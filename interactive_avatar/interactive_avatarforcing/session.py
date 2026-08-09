from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput

from .config import InteractiveAvatarForcingConfig

if TYPE_CHECKING:
    from .runtime import InteractiveAvatarForcingRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairedFrame:
    video_rgb: np.ndarray
    audio_s16le: bytes
    epoch: int
    final_segment_frame: bool = False


class LiveFaceCropper:
    def __init__(self, size: int) -> None:
        self.size = size
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._bounds: tuple[int, int, int, int] | None = None
        self._frame_index = 0

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        height, width = frame_rgb.shape[:2]
        if self._frame_index % 10 == 0:
            scale = min(1.0, 360.0 / height)
            small = cv2.resize(frame_rgb, (round(width * scale), round(height * scale)))
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            faces = self._detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            if len(faces):
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                self._bounds = tuple(round(value / scale) for value in (x, y, w, h))
            else:
                self._bounds = None
        self._frame_index += 1

        if self._bounds is None:
            return None
        x, y, w, h = self._bounds
        center_x, center_y = x + w // 2, y + h // 2
        radius = max(w, h)
        radius = min(radius, center_x, width - center_x, center_y, height - center_y)
        if radius <= 0:
            return cv2.resize(frame_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        face = frame_rgb[
            center_y - radius : center_y + radius,
            center_x - radius : center_x + radius,
        ]
        return cv2.resize(face, (self.size, self.size), interpolation=cv2.INTER_AREA)


class InteractiveAvatarForcingLiveAvatar:
    """Continuously publish generated speaking, listening, and idle motion."""

    def __init__(
        self,
        runtime: InteractiveAvatarForcingRuntime,
        config: InteractiveAvatarForcingConfig,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.audio_output = QueueAudioOutput(
            sample_rate=config.sample_rate,
            wait_playback_start=True,
        )
        self._media: asyncio.Queue[PairedFrame] = asyncio.Queue(
            maxsize=config.output_queue_frames
        )
        self._speech = bytearray()
        self._listen = bytearray()
        self._camera: deque[np.ndarray] = deque(maxlen=config.history_frames)
        self._camera_updated_at: float | None = None
        self._epoch = 0
        self._segment_active = False
        self._segment_ended = False
        self._segment_samples = 0
        self._published_speech_samples = 0
        self._playback_started = False
        self._closed = False
        self._room: rtc.Room | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._video_source: rtc.VideoSource | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._listener_tasks: list[asyncio.Task[None]] = []
        self._listener_identity: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._control_lock = asyncio.Lock()
        self.audio_output.on("clear_buffer", self._on_clear_buffer)

    async def start(self, room: rtc.Room) -> None:
        if self._tasks:
            return
        self._loop = asyncio.get_running_loop()
        self._room = room
        self._audio_source = rtc.AudioSource(self.config.sample_rate, 1, queue_size_ms=120)
        self._video_source = rtc.VideoSource(self.config.width, self.config.height)
        await room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track(
                "interactive-avatarforcing-audio", self._audio_source
            ),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await room.local_participant.publish_track(
            rtc.LocalVideoTrack.create_video_track(
                "interactive-avatarforcing-video", self._video_source
            ),
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                simulcast=False,
                video_encoding=rtc.VideoEncoding(
                    max_bitrate=2_500_000,
                    max_framerate=self.config.fps,
                ),
                degradation_preference=rtc.DegradationPreference.MAINTAIN_FRAMERATE_AND_RESOLUTION,
            ),
        )

        room.on("participant_connected", self._on_participant_connected)
        room.on("participant_disconnected", self._on_participant_disconnected)
        for participant in room.remote_participants.values():
            self._attach_listener(participant)
            break
        self._tasks = [
            asyncio.create_task(self._consume_tts(), name="interactive-af-tts"),
            asyncio.create_task(self._render_loop(), name="interactive-af-render"),
            asyncio.create_task(self._publish_loop(), name="interactive-af-publish"),
        ]
        await self._set_state("idle")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio_output.off("clear_buffer", self._on_clear_buffer)
        if self._room:
            self._room.off("participant_connected", self._on_participant_connected)
            self._room.off("participant_disconnected", self._on_participant_disconnected)
        for task in [*self._tasks, *self._listener_tasks]:
            task.cancel()
        await asyncio.gather(*self._tasks, *self._listener_tasks, return_exceptions=True)
        await self.audio_output.aclose()
        if self._audio_source:
            await self._audio_source.aclose()
        if self._video_source:
            await self._video_source.aclose()

    async def _consume_tts(self) -> None:
        async for item in self.audio_output:
            async with self._control_lock:
                if isinstance(item, AudioSegmentEnd):
                    self._segment_ended = True
                    continue
                if item.sample_rate != self.config.sample_rate or item.num_channels != 1:
                    raise RuntimeError(
                        f"unexpected TTS format: {item.sample_rate} Hz, {item.num_channels} channels"
                    )
                if not self._segment_active:
                    await self._begin_segment()
                self._speech.extend(bytes(item.data))
                self._segment_samples += item.samples_per_channel

    async def _begin_segment(self) -> None:
        self._epoch += 1
        self._drain_media()
        if self._audio_source:
            self._audio_source.clear_queue()
        self._speech.clear()
        self._segment_active = True
        self._segment_ended = False
        self._segment_samples = 0
        self._published_speech_samples = 0
        self._playback_started = False
        await self._set_state("preparing")

    async def _render_loop(self) -> None:
        silence = bytes(self.config.bytes_per_block)
        while True:
            while self._media.qsize() >= self.config.block_frames:
                await asyncio.sleep(0.01)
            async with self._control_lock:
                epoch = self._epoch
                consumed_speech = bytes(self._speech[: self.config.bytes_per_block])
                speech = consumed_speech.ljust(
                    self.config.bytes_per_block, b"\0"
                )
                del self._speech[: self.config.bytes_per_block]
                consumed_listen = bytes(self._listen[: self.config.bytes_per_block])
                listen = consumed_listen.ljust(
                    self.config.bytes_per_block, b"\0"
                )
                del self._listen[: self.config.bytes_per_block]
                camera = self._fresh_camera_frames()
                final_chunk = self._segment_active and self._segment_ended and not self._speech
                segment_active = self._segment_active

            snapshot = await asyncio.to_thread(self.runtime.snapshot)
            try:
                result = await asyncio.to_thread(
                    self.runtime.generate_block,
                    speech if segment_active else silence,
                    listen,
                    camera,
                )
            except Exception:
                logger.exception("interactive AvatarForcing block failed")
                await asyncio.to_thread(self.runtime.restore, snapshot)
                async with self._control_lock:
                    self._restore_consumed_audio(epoch, consumed_speech, consumed_listen)
                await asyncio.sleep(0.1)
                continue
            logger.debug("interactive AvatarForcing block %.3fs", result.total_seconds)

            async with self._control_lock:
                if epoch != self._epoch:
                    await asyncio.to_thread(self.runtime.restore, snapshot)
                    continue
                for index, frame in enumerate(result.frames_rgb):
                    start = index * self.config.samples_per_frame * 2
                    end = start + self.config.samples_per_frame * 2
                    await self._media.put(
                        PairedFrame(
                            frame,
                            speech[start:end],
                            epoch,
                            final_segment_frame=(
                                final_chunk and index == self.config.block_frames - 1
                            ),
                        )
                    )

    async def _publish_loop(self) -> None:
        assert self._audio_source is not None and self._video_source is not None
        loop = asyncio.get_running_loop()
        next_frame_at = loop.time()
        while True:
            frame = await self._next_current_frame()
            delay = next_frame_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            next_frame_at = max(next_frame_at + 1.0 / self.config.fps, loop.time() - 0.1)

            speaking = bool(np.frombuffer(frame.audio_s16le, dtype=np.int16).any())
            if speaking:
                self._published_speech_samples += self.config.samples_per_frame
                if not self._playback_started:
                    self._playback_started = True
                    self.audio_output.notify_playback_started()
                    await self._set_state("talking")

            await self._audio_source.capture_frame(
                rtc.AudioFrame(
                    data=frame.audio_s16le,
                    sample_rate=self.config.sample_rate,
                    num_channels=1,
                    samples_per_channel=self.config.samples_per_frame,
                )
            )
            self._video_source.capture_frame(
                rtc.VideoFrame(
                    width=self.config.width,
                    height=self.config.height,
                    type=rtc.VideoBufferType.RGB24,
                    data=frame.video_rgb.tobytes(),
                ),
                timestamp_us=time.time_ns() // 1000,
            )
            if frame.final_segment_frame:
                self.audio_output.notify_playback_finished(
                    self._segment_samples / self.config.sample_rate,
                    False,
                )
                self._segment_active = False
                self._segment_ended = False
                await self._set_state("idle")

    async def _next_current_frame(self) -> PairedFrame:
        while True:
            frame = await self._media.get()
            if frame.epoch == self._epoch:
                return frame

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        self._attach_listener(participant)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity != self._listener_identity:
            return
        for task in self._listener_tasks:
            task.cancel()
        self._listener_tasks = []
        self._listener_identity = None
        self._listen.clear()
        self._camera.clear()
        self._camera_updated_at = None

    def _attach_listener(self, participant: rtc.RemoteParticipant) -> None:
        if self._listener_identity is not None:
            return
        self._listener_identity = participant.identity
        self._listener_tasks = [
            asyncio.create_task(
                self._consume_listener_audio(participant),
                name=f"interactive-af-audio-{participant.identity}",
            ),
            asyncio.create_task(
                self._consume_listener_video(participant),
                name=f"interactive-af-video-{participant.identity}",
            ),
        ]

    async def _consume_listener_audio(self, participant: rtc.RemoteParticipant) -> None:
        stream = rtc.AudioStream.from_participant(
            participant=participant,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            sample_rate=self.config.sample_rate,
            num_channels=1,
            frame_size_ms=40,
            capacity=20,
        )
        try:
            async for event in stream:
                async with self._control_lock:
                    self._listen.extend(bytes(event.frame.data))
                    max_bytes = self.config.history_bytes * 2
                    if len(self._listen) > max_bytes:
                        del self._listen[: len(self._listen) - max_bytes]
        finally:
            await stream.aclose()

    async def _consume_listener_video(self, participant: rtc.RemoteParticipant) -> None:
        stream = rtc.VideoStream.from_participant(
            participant=participant,
            track_source=rtc.TrackSource.SOURCE_CAMERA,
            format=rtc.VideoBufferType.RGB24,
            capacity=2,
        )
        cropper = LiveFaceCropper(self.config.width)
        try:
            async for event in stream:
                frame = event.frame
                rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                    frame.height, frame.width, 3
                )
                cropped = cropper.crop(rgb)
                async with self._control_lock:
                    if cropped is None:
                        self._camera.clear()
                        self._camera_updated_at = None
                    else:
                        self._camera.append(cropped)
                        self._camera_updated_at = time.monotonic()
        finally:
            await stream.aclose()

    def _on_clear_buffer(self) -> None:
        if self._closed or self._loop is None:
            return
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._handle_interruption())
        )

    async def _handle_interruption(self) -> None:
        async with self._control_lock:
            self._epoch += 1
            self._speech.clear()
            self._segment_ended = False
            while not self._media.empty():
                self._media.get_nowait()
            if self._audio_source:
                self._audio_source.clear_queue()
            if self._segment_active:
                self.audio_output.notify_playback_finished(
                    self._published_speech_samples / self.config.sample_rate,
                    True,
                )
            self._segment_active = False
            await self._set_state("interrupted")

    def _restore_consumed_audio(self, epoch: int, speech: bytes, listen: bytes) -> None:
        if epoch != self._epoch:
            return
        self._speech[:0] = speech
        self._listen[:0] = listen

    def _fresh_camera_frames(self) -> list[np.ndarray]:
        if (
            self._camera_updated_at is None
            or time.monotonic() - self._camera_updated_at > self.config.camera_stale_seconds
        ):
            return []
        return list(self._camera)[-self.config.block_frames :]

    def _drain_media(self) -> None:
        while not self._media.empty():
            self._media.get_nowait()

    async def _set_state(self, state: str) -> None:
        if self._room and self._room.isconnected():
            await self._room.local_participant.set_attributes(
                {
                    "interactive_avatarforcing.state": state,
                    "interactive_avatarforcing.generation_id": str(self._epoch),
                }
            )
