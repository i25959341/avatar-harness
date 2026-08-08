from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput

from .config import FlashHeadConfig
from .runtime import FlashHeadRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairedFrame:
    video_rgb: np.ndarray
    audio_s16le: bytes
    epoch: int
    speech_audio: bool


@dataclass(frozen=True)
class SegmentComplete:
    epoch: int
    playback_position: float
    interrupted: bool = False


class FlashHeadLiveAvatar:
    """Route AgentSession TTS through FlashHead and publish paired RTC media."""

    def __init__(self, runtime: FlashHeadRuntime, config: FlashHeadConfig) -> None:
        self.runtime = runtime
        self.config = config
        self.audio_output = QueueAudioOutput(
            sample_rate=config.sample_rate,
            wait_playback_start=True,
        )
        self._idle_frames = self._load_idle_frames()
        self._idle_index = 0
        self._published_history: deque[np.ndarray] = deque(maxlen=config.motion_frames)
        self._media: asyncio.Queue[PairedFrame | SegmentComplete] = asyncio.Queue(
            maxsize=config.output_queue_frames
        )
        self._epoch = 0
        self._closed = False
        self._room: rtc.Room | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._video_source: rtc.VideoSource | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._transition_task: asyncio.Task[None] | None = None
        self._interruption_hold_frame: np.ndarray | None = None
        self._segment_buffer = bytearray()
        self._segment_active = False
        self._segment_speech_samples = 0
        self._segment_first_chunk = False
        self._playback_started = False
        self.audio_output.on("clear_buffer", self._on_clear_buffer)

    async def start(self, room: rtc.Room) -> None:
        if self._tasks:
            return
        self._room = room
        self._audio_source = rtc.AudioSource(
            self.config.sample_rate,
            1,
            queue_size_ms=120,
        )
        self._video_source = rtc.VideoSource(self.config.width, self.config.height)
        audio_track = rtc.LocalAudioTrack.create_audio_track("flashhead-audio", self._audio_source)
        video_track = rtc.LocalVideoTrack.create_video_track("flashhead-video", self._video_source)
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
        self._tasks = [
            asyncio.create_task(self._consume_audio(), name="flashhead-audio-consumer"),
            asyncio.create_task(self._publish(), name="flashhead-publisher"),
        ]
        await self._set_state("idle")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio_output.off("clear_buffer", self._on_clear_buffer)
        if self._transition_task:
            self._transition_task.cancel()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._transition_task:
            await asyncio.gather(self._transition_task, return_exceptions=True)
        await self.audio_output.aclose()
        if self._audio_source:
            await self._audio_source.aclose()
        if self._video_source:
            await self._video_source.aclose()

    async def _consume_audio(self) -> None:
        async for item in self.audio_output:
            if isinstance(item, AudioSegmentEnd):
                await self._finish_segment()
                continue
            if item.sample_rate != self.config.sample_rate or item.num_channels != 1:
                raise RuntimeError(
                    f"unexpected TTS format: {item.sample_rate} Hz, {item.num_channels} channels"
                )
            if not self._segment_active:
                await self._begin_segment()
            pcm = bytes(item.data)
            self._segment_buffer.extend(pcm)
            self._segment_speech_samples += item.samples_per_channel
            await self._generate_full_chunks()

    async def _begin_segment(self) -> None:
        if self._transition_task:
            await asyncio.gather(self._transition_task, return_exceptions=True)
            self._transition_task = None
        self._interruption_hold_frame = None
        self._epoch += 1
        self._segment_active = True
        self._segment_first_chunk = True
        self._segment_speech_samples = 0
        self._playback_started = False
        prefix_bytes = self.config.bridge_frames * self.config.samples_per_frame * 2
        self._segment_buffer = bytearray(prefix_bytes)
        await self._set_state("preparing")

    async def _generate_full_chunks(self) -> None:
        while len(self._segment_buffer) >= self.config.bytes_per_chunk:
            chunk = bytes(self._segment_buffer[: self.config.bytes_per_chunk])
            del self._segment_buffer[: self.config.bytes_per_chunk]
            pairs = await self._generate_pairs(chunk, self.config.chunk_frames)
            await self._enqueue_pairs(pairs)

    async def _finish_segment(self) -> None:
        if not self._segment_active:
            return
        suffix_bytes = self.config.bridge_frames * self.config.samples_per_frame * 2
        self._segment_buffer.extend(bytes(suffix_bytes))
        final_pairs: list[PairedFrame] = []
        while self._segment_buffer:
            valid_samples = min(
                len(self._segment_buffer) // 2,
                self.config.samples_per_chunk,
            )
            chunk_bytes = min(len(self._segment_buffer), self.config.bytes_per_chunk)
            chunk = bytes(self._segment_buffer[:chunk_bytes]).ljust(
                self.config.bytes_per_chunk, b"\0"
            )
            del self._segment_buffer[:chunk_bytes]
            valid_frames = math.ceil(valid_samples / self.config.samples_per_frame)
            final_pairs.extend(await self._generate_pairs(chunk, valid_frames))

        if final_pairs:
            self._blend_to_idle(final_pairs)
            await self._enqueue_pairs(final_pairs)
        duration = self._segment_speech_samples / self.config.sample_rate
        await self._media.put(SegmentComplete(self._epoch, duration))
        self._segment_active = False
        self._segment_buffer.clear()

    async def _generate_pairs(self, chunk: bytes, valid_frames: int) -> list[PairedFrame]:
        epoch = self._epoch
        first_chunk = self._segment_first_chunk
        self._segment_first_chunk = False
        if first_chunk:
            motion = list(self._published_history)
            if not motion:
                motion = [self._idle_frames[self._idle_index]]
            frames = await asyncio.to_thread(
                self.runtime.generate_from_motion,
                motion,
                chunk,
            )
        else:
            frames = await asyncio.to_thread(self.runtime.generate_chunk, chunk)
        if epoch != self._epoch:
            return []

        pairs: list[PairedFrame] = []
        for index in range(valid_frames):
            start = index * self.config.samples_per_frame * 2
            audio = chunk[start : start + self.config.samples_per_frame * 2].ljust(
                self.config.samples_per_frame * 2, b"\0"
            )
            frame = frames[index].copy()
            if first_chunk and index < self.config.bridge_frames // 2:
                source = self._published_history[-1] if self._published_history else frame
                alpha = (index + 1) / (self.config.bridge_frames // 2)
                frame = cv2.addWeighted(source, 1.0 - alpha, frame, alpha, 0)
            speech_audio = first_chunk and index >= self.config.bridge_frames
            if not first_chunk:
                speech_audio = bool(np.frombuffer(audio, dtype=np.int16).any())
            pairs.append(PairedFrame(frame, audio, epoch, speech_audio))
        return pairs

    async def _enqueue_pairs(self, pairs: list[PairedFrame]) -> None:
        for pair in pairs:
            if pair.epoch == self._epoch:
                await self._media.put(pair)

    def _blend_to_idle(self, pairs: list[PairedFrame]) -> None:
        bridge_count = min(self.config.bridge_frames, len(pairs))
        if not bridge_count:
            return
        target_end = self._nearest_idle_index(pairs[-1].video_rgb)
        target_start = (target_end - bridge_count + 1) % len(self._idle_frames)
        blend_count = min(self.config.bridge_frames // 2, bridge_count)
        for step in range(blend_count):
            pair_index = len(pairs) - blend_count + step
            idle_index = (target_end - blend_count + 1 + step) % len(self._idle_frames)
            alpha = (step + 1) / blend_count
            pair = pairs[pair_index]
            pairs[pair_index] = PairedFrame(
                cv2.addWeighted(
                    pair.video_rgb,
                    1.0 - alpha,
                    self._idle_frames[idle_index],
                    alpha,
                    0,
                ),
                pair.audio_s16le,
                pair.epoch,
                False,
            )
        self._idle_index = (target_start + bridge_count) % len(self._idle_frames)

    async def _publish(self) -> None:
        assert self._audio_source is not None and self._video_source is not None
        loop = asyncio.get_running_loop()
        next_frame_at = loop.time()
        silence = bytes(self.config.samples_per_frame * 2)
        while True:
            delay = next_frame_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            next_frame_at += 1.0 / self.config.fps
            if next_frame_at < loop.time() - 0.5:
                next_frame_at = loop.time()

            item: PairedFrame | None = None
            while not self._media.empty():
                queued = self._media.get_nowait()
                if isinstance(queued, SegmentComplete):
                    if queued.epoch == self._epoch:
                        self._interruption_hold_frame = None
                        self.audio_output.notify_playback_finished(
                            queued.playback_position,
                            queued.interrupted,
                        )
                        await self._set_state("idle")
                    continue
                if queued.epoch == self._epoch:
                    item = queued
                    break

            if item is None:
                holding = (
                    self._interruption_hold_frame is not None
                    and self._transition_task is not None
                    and not self._transition_task.done()
                )
                if holding:
                    video = self._interruption_hold_frame
                else:
                    video = self._idle_frames[self._idle_index]
                    self._idle_index = (self._idle_index + 1) % len(self._idle_frames)
                audio = silence
            else:
                video = item.video_rgb
                audio = item.audio_s16le
                if item.speech_audio and not self._playback_started:
                    self._playback_started = True
                    self.audio_output.notify_playback_started()
                    await self._set_state("talking")

            self._published_history.append(video.copy())
            await self._audio_source.capture_frame(
                rtc.AudioFrame(
                    data=audio,
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
                    data=video.tobytes(),
                ),
                timestamp_us=time.time_ns() // 1000,
            )

    def _on_clear_buffer(self) -> None:
        if self._closed:
            return
        self._epoch += 1
        self._segment_active = False
        self._segment_buffer.clear()
        self._drain_media_queue()
        if self._audio_source:
            self._audio_source.clear_queue()
        history = [frame.copy() for frame in self._published_history]
        if self._transition_task and not self._transition_task.done():
            self._transition_task.cancel()
        mode = self.config.interruption_transition
        self._interruption_hold_frame = history[-1].copy() if history and mode == "vae" else None
        handler = {
            "generated": self._generate_interruption_bridge,
            "vae": self._generate_vae_interruption_bridge,
            "pixel": self._generate_pixel_interruption_bridge,
            "hard": self._generate_hard_interruption,
        }[mode]
        self._transition_task = asyncio.create_task(
            handler(self._epoch, history),
            name=f"flashhead-{mode}-interruption",
        )

    async def _generate_interruption_bridge(self, epoch: int, history: list[np.ndarray]) -> None:
        if not history:
            return
        chunk = bytes(self.config.bytes_per_chunk)
        frames = await asyncio.to_thread(
            self.runtime.generate_from_motion,
            history,
            chunk,
        )
        if epoch != self._epoch:
            return
        pairs = [
            PairedFrame(
                frames[index].copy(),
                bytes(self.config.samples_per_frame * 2),
                epoch,
                False,
            )
            for index in range(self.config.bridge_frames)
        ]
        self._blend_to_idle(pairs)
        await self._enqueue_pairs(pairs)
        await self._media.put(SegmentComplete(epoch, 0.0, interrupted=True))

    async def _generate_vae_interruption_bridge(
        self, epoch: int, history: list[np.ndarray]
    ) -> None:
        if not history:
            await self._media.put(SegmentComplete(epoch, 0.0, interrupted=True))
            return
        resume_index = self._nearest_idle_index(history[-1])
        target = self._idle_window_ending_at(resume_index)
        started = time.perf_counter()
        try:
            frames = await asyncio.to_thread(
                self.runtime.generate_vae_transition,
                history,
                target,
                self.config.interruption_bridge_frames,
            )
        except Exception:
            logger.exception("VAE interruption bridge failed; using pixel fallback")
            if epoch == self._epoch:
                await self._generate_pixel_interruption_bridge(epoch, history)
            return
        if epoch != self._epoch:
            return
        logger.info(
            "VAE interruption bridge generated in %.0fms",
            (time.perf_counter() - started) * 1000,
        )
        silence = bytes(self.config.samples_per_frame * 2)
        pairs = [PairedFrame(frame.copy(), silence, epoch, False) for frame in frames]
        self._idle_index = resume_index
        await self._enqueue_pairs(pairs)
        await self._media.put(SegmentComplete(epoch, 0.0, interrupted=True))

    async def _generate_pixel_interruption_bridge(
        self, epoch: int, history: list[np.ndarray]
    ) -> None:
        if not history:
            await self._media.put(SegmentComplete(epoch, 0.0, interrupted=True))
            return
        count = self.config.interruption_bridge_frames
        resume_index = self._nearest_idle_index(history[-1])
        silence = bytes(self.config.samples_per_frame * 2)
        pairs: list[PairedFrame] = []
        for step in range(count):
            progress = (step + 1) / count
            alpha = progress * progress * (3.0 - 2.0 * progress)
            target = self._idle_frames[(resume_index - count + step) % len(self._idle_frames)]
            frame = cv2.addWeighted(history[-1], 1.0 - alpha, target, alpha, 0)
            pairs.append(PairedFrame(frame, silence, epoch, False))
        if epoch != self._epoch:
            return
        self._idle_index = resume_index
        await self._enqueue_pairs(pairs)
        await self._media.put(SegmentComplete(epoch, 0.0, interrupted=True))

    async def _generate_hard_interruption(self, epoch: int, history: list[np.ndarray]) -> None:
        if epoch != self._epoch:
            return
        if history:
            self._idle_index = self._nearest_idle_index(history[-1])
        await self._media.put(SegmentComplete(epoch, 0.0, interrupted=True))

    def _drain_media_queue(self) -> None:
        while not self._media.empty():
            try:
                self._media.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _nearest_idle_index(self, frame: np.ndarray) -> int:
        target = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            (64, 64),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        return min(
            range(len(self._idle_frames)),
            key=lambda index: float(
                np.mean(
                    np.square(
                        cv2.resize(
                            cv2.cvtColor(self._idle_frames[index], cv2.COLOR_RGB2GRAY),
                            (64, 64),
                            interpolation=cv2.INTER_AREA,
                        ).astype(np.float32)
                        - target
                    )
                )
            ),
        )

    def _idle_window_ending_at(self, end_index: int) -> list[np.ndarray]:
        start = end_index - self.config.motion_frames
        return [
            self._idle_frames[(start + offset) % len(self._idle_frames)]
            for offset in range(self.config.motion_frames)
        ]

    def _load_idle_frames(self) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(self.config.idle_video))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(
                    frame,
                    (self.config.width, self.config.height),
                    interpolation=cv2.INTER_AREA,
                )
                source.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if not source or source_fps <= 0:
            raise RuntimeError(f"could not decode idle video: {self.config.idle_video}")
        duration = len(source) / source_fps
        count = round(duration * self.config.fps)
        return [
            source[round(((index / self.config.fps) % duration) * source_fps) % len(source)]
            for index in range(count)
        ]

    async def _set_state(self, state: str) -> None:
        if self._room and self._room.isconnected():
            await self._room.local_participant.set_attributes(
                {
                    "flashhead.state": state,
                    "flashhead.generation_id": str(self._epoch),
                }
            )
