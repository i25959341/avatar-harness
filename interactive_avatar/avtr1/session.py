from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput

from .client import Avtr1RendererClient
from .config import Avtr1Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairedFrame:
    video_i420: bytes
    audio_s16le: bytes
    epoch: int
    final_segment_frame: bool = False


class Avtr1LiveAvatar:
    """Publish one continuous AVTR-1 speaking, listening, and idle state."""

    def __init__(self, client: Avtr1RendererClient, config: Avtr1Config) -> None:
        self.client = client
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
        self._state: bytes | None = None
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
        self._listen_task: asyncio.Task[None] | None = None
        self._listener_identity: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._control_lock = asyncio.Lock()
        self._speech_changed = asyncio.Event()
        self.audio_output.on("clear_buffer", self._on_clear_buffer)

    async def start(self, room: rtc.Room) -> None:
        if self._tasks:
            return
        self._loop = asyncio.get_running_loop()
        self._room = room
        await self.client.validate()

        silence = bytes(self.window_bytes)
        warm = await self.client.render(silence, silence, None)
        self._state = warm.state
        for frame in warm.frames_i420:
            await self._media.put(
                PairedFrame(frame, bytes(self.config.bytes_per_frame), self._epoch)
            )

        self._audio_source = rtc.AudioSource(
            self.config.sample_rate,
            1,
            queue_size_ms=120,
        )
        self._video_source = rtc.VideoSource(self.config.width, self.config.height)
        audio_track = rtc.LocalAudioTrack.create_audio_track("avtr1-audio", self._audio_source)
        video_track = rtc.LocalVideoTrack.create_video_track("avtr1-video", self._video_source)
        await room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await room.local_participant.publish_track(
            video_track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                simulcast=False,
                video_encoding=rtc.VideoEncoding(
                    max_bitrate=3_000_000,
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
            asyncio.create_task(self._consume_tts(), name="avtr1-tts-consumer"),
            asyncio.create_task(self._render_loop(), name="avtr1-renderer"),
            asyncio.create_task(self._publish_loop(), name="avtr1-publisher"),
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
        if self._listen_task:
            self._listen_task.cancel()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(
            *self._tasks,
            *([self._listen_task] if self._listen_task else []),
            return_exceptions=True,
        )
        await self.client.aclose()
        await self.audio_output.aclose()
        if self._audio_source:
            await self._audio_source.aclose()
        if self._video_source:
            await self._video_source.aclose()

    @property
    def window_bytes(self) -> int:
        return (self.config.current_samples + self.config.future_samples) * 2

    async def _consume_tts(self) -> None:
        async for item in self.audio_output:
            async with self._control_lock:
                if isinstance(item, AudioSegmentEnd):
                    self._segment_ended = True
                    self._speech_changed.set()
                    continue
                if item.sample_rate != self.config.sample_rate or item.num_channels != 1:
                    raise RuntimeError(
                        f"unexpected TTS format: {item.sample_rate} Hz, {item.num_channels} channels"
                    )
                if not self._segment_active:
                    self._epoch += 1
                    self._drain_media()
                    self._speech.clear()
                    self._segment_active = True
                    self._segment_ended = False
                    self._segment_samples = 0
                    self._published_speech_samples = 0
                    self._playback_started = False
                    await self._set_state("preparing")
                self._speech.extend(bytes(item.data))
                self._segment_samples += item.samples_per_channel
                self._speech_changed.set()

    async def _render_loop(self) -> None:
        while True:
            while self._media.qsize() >= self.config.chunk_frames:
                await asyncio.sleep(0.01)

            await self._wait_for_speech_lookahead()
            async with self._control_lock:
                epoch = self._epoch
                current_bytes = self.config.current_samples * 2
                consumed_speech = bytes(self._speech[:current_bytes])
                consumed_listen = bytes(self._listen[:current_bytes])
                speech_window = bytes(self._speech[: self.window_bytes]).ljust(
                    self.window_bytes, b"\0"
                )
                listen_window = bytes(self._listen[: self.window_bytes]).ljust(
                    self.window_bytes, b"\0"
                )
                current_audio = consumed_speech.ljust(
                    current_bytes, b"\0"
                )
                del self._speech[:current_bytes]
                del self._listen[:current_bytes]
                final_chunk = self._segment_active and self._segment_ended and not self._speech
                state = self._state

            try:
                rendered = await self.client.render(speech_window, listen_window, state)
            except Exception:
                logger.exception("AVTR-1 render call failed; retrying")
                async with self._control_lock:
                    self._restore_consumed_audio(epoch, consumed_speech, consumed_listen)
                await asyncio.sleep(0.1)
                continue

            async with self._control_lock:
                if epoch != self._epoch:
                    continue
                self._state = rendered.state
                for index, frame in enumerate(rendered.frames_i420):
                    start = index * self.config.bytes_per_frame
                    audio = current_audio[start : start + self.config.bytes_per_frame]
                    await self._media.put(
                        PairedFrame(
                            frame,
                            audio,
                            epoch,
                            final_segment_frame=(
                                final_chunk and index == len(rendered.frames_i420) - 1
                            ),
                        )
                    )

    async def _wait_for_speech_lookahead(self) -> None:
        if not self._segment_active or self._segment_ended or len(self._speech) >= self.window_bytes:
            return
        self._speech_changed.clear()
        try:
            await asyncio.wait_for(self._speech_changed.wait(), timeout=0.18)
        except asyncio.TimeoutError:
            pass

    async def _publish_loop(self) -> None:
        assert self._audio_source is not None and self._video_source is not None
        loop = asyncio.get_running_loop()
        next_frame_at = loop.time()
        while True:
            delay = next_frame_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            next_frame_at += 1.0 / self.config.fps
            if next_frame_at < loop.time() - 0.5:
                next_frame_at = loop.time()

            frame = await self._next_current_frame()
            if frame is None:
                continue
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
                    type=rtc.VideoBufferType.I420,
                    data=frame.video_i420,
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

    async def _next_current_frame(self) -> PairedFrame | None:
        while True:
            frame = await self._media.get()
            if frame.epoch == self._epoch:
                return frame

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        self._attach_listener(participant)

    def _attach_listener(self, participant: rtc.RemoteParticipant) -> None:
        if self._listen_task and not self._listen_task.done():
            return
        self._listener_identity = participant.identity
        self._listen_task = asyncio.create_task(
            self._consume_listener(participant),
            name=f"avtr1-listener-{participant.identity}",
        )

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity != self._listener_identity:
            return
        if self._listen_task:
            self._listen_task.cancel()
        self._listen_task = None
        self._listener_identity = None
        self._listen.clear()

    async def _consume_listener(self, participant: rtc.RemoteParticipant) -> None:
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
                    if len(self._listen) > self.window_bytes * 3:
                        del self._listen[: len(self._listen) - self.window_bytes * 3]
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
            self._drain_media()
            self._segment_ended = False
            if self._audio_source:
                self._audio_source.clear_queue()
            if self._segment_active:
                self.audio_output.notify_playback_finished(
                    self._published_speech_samples / self.config.sample_rate,
                    True,
                )
            self._segment_active = False
            await self._set_state("interrupted")

    def _drain_media(self) -> None:
        while not self._media.empty():
            self._media.get_nowait()

    def _restore_consumed_audio(self, epoch: int, speech: bytes, listen: bytes) -> None:
        if epoch != self._epoch:
            return
        self._speech[:0] = speech
        self._listen[:0] = listen

    async def _set_state(self, state: str) -> None:
        if self._room and self._room.isconnected():
            await self._room.local_participant.set_attributes(
                {"avtr1.state": state, "avtr1.generation_id": str(self._epoch)}
            )
