from __future__ import annotations

import asyncio
import logging
import queue
import time

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput

from interactive_avatar.events import FrameType, OutputFrame
from interactive_avatar.livekit_generator import AvatarHarnessGenerator, AvatarHarnessOptions

from .config import IMTalkerConfig

logger = logging.getLogger(__name__)


class IMTalkerLiveAvatar:
    """Publish IMTalker frames and their paired PCM on one LiveKit clock."""

    def __init__(
        self,
        config: IMTalkerConfig,
        *,
        preloaded_agent: object | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.audio_output = QueueAudioOutput(
            sample_rate=config.sample_rate,
            wait_playback_start=True,
        )
        self.generator = AvatarHarnessGenerator(
            source_image_path=str(config.source_image),
            idle_cache_path=str(config.idle_cache),
            options=AvatarHarnessOptions(
                video_width=config.width,
                video_height=config.height,
                video_fps=config.fps,
                audio_sample_rate=config.sample_rate,
                min_chunk_frames=config.min_chunk_frames,
                max_chunk_frames=config.max_chunk_frames,
                default_chunk_frames=config.default_chunk_frames,
            ),
            preloaded_agent=preloaded_agent,
        )
        self._room: rtc.Room | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._video_source: rtc.VideoSource | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._control_lock = asyncio.Lock()
        self._closed = False
        self._segment_active = False
        self._playback_started = False
        self._playback_finished = False
        self._published_speech_samples = 0
        self.audio_output.on("clear_buffer", self._on_clear_buffer)

    async def start(self, room: rtc.Room) -> None:
        if self._tasks:
            return
        self._loop = asyncio.get_running_loop()
        self._room = room
        await self.generator.start()

        self._audio_source = rtc.AudioSource(
            self.config.sample_rate,
            1,
            queue_size_ms=120,
        )
        self._video_source = rtc.VideoSource(self.config.width, self.config.height)
        audio_track = rtc.LocalAudioTrack.create_audio_track("imtalker-audio", self._audio_source)
        video_track = rtc.LocalVideoTrack.create_video_track("imtalker-video", self._video_source)
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
            asyncio.create_task(self._consume_audio(), name="imtalker-audio-consumer"),
            asyncio.create_task(self._publish(), name="imtalker-publisher"),
        ]
        await self._set_state("idle")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.audio_output.off("clear_buffer", self._on_clear_buffer)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.generator.stop()
        await self.audio_output.aclose()
        if self._audio_source:
            await self._audio_source.aclose()
        if self._video_source:
            await self._video_source.aclose()

    async def _consume_audio(self) -> None:
        async for item in self.audio_output:
            async with self._control_lock:
                if not isinstance(item, AudioSegmentEnd) and not self._segment_active:
                    self._segment_active = True
                    self._playback_started = False
                    self._playback_finished = False
                    self._published_speech_samples = 0
                    await self._set_state("preparing")
                await self.generator.push_audio(item)

    async def _publish(self) -> None:
        assert self._audio_source is not None and self._video_source is not None
        loop = asyncio.get_running_loop()
        next_frame_at = loop.time()
        silence = bytes(self.config.bytes_per_frame)

        while True:
            delay = next_frame_at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            next_frame_at += 1.0 / self.config.fps
            if next_frame_at < loop.time() - 0.5:
                next_frame_at = loop.time()

            try:
                frame = self.generator.get_output_frame_nowait()
            except queue.Empty:
                frame = self.generator.get_idle_fallback()

            speaking = frame.type == FrameType.SPEAKING
            audio = self._paired_audio(frame) if speaking else silence
            if speaking:
                self._published_speech_samples += self.config.samples_per_frame
                if not self._playback_started:
                    self._playback_started = True
                    self.audio_output.notify_playback_started()
                    await self._set_state("talking")

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
                    data=frame.video_frame.tobytes(),
                ),
                timestamp_us=time.time_ns() // 1000,
            )

            if frame.final_chunk:
                self._notify_playback_finished(interrupted=False)
                await self._set_state("transitioning")
            elif frame.type == FrameType.IDLE and self._playback_finished:
                self._segment_active = False
                await self._set_state("idle")

    def _paired_audio(self, frame: OutputFrame) -> bytes:
        if frame.audio_frame is None:
            return bytes(self.config.bytes_per_frame)
        return bytes(frame.audio_frame[: self.config.bytes_per_frame]).ljust(
            self.config.bytes_per_frame, b"\0"
        )

    def _notify_playback_finished(self, *, interrupted: bool) -> None:
        if self._playback_finished:
            return
        self._playback_finished = True
        position = self._published_speech_samples / self.config.sample_rate
        self.audio_output.notify_playback_finished(position, interrupted)

    def _on_clear_buffer(self) -> None:
        if self._closed or self._loop is None:
            return
        self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._handle_interruption()))

    async def _handle_interruption(self) -> None:
        async with self._control_lock:
            if self._audio_source:
                self._audio_source.clear_queue()
            if self._segment_active:
                self._notify_playback_finished(interrupted=True)
            await self._set_state("interrupted")
            await asyncio.to_thread(self.generator.clear_buffer)
            self._segment_active = False

    async def _set_state(self, state: str) -> None:
        if self._room and self._room.isconnected():
            await self._room.local_participant.set_attributes({"imtalker.state": state})
