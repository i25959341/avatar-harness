"""
TalkBox LiveKit VideoGenerator implementation.

Integrates TalkBox with LiveKit Agents SDK for real-time WebRTC streaming.
Implements the VideoGenerator interface to work with AvatarRunner.
"""

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import torch
from PIL import Image

# LiveKit imports (optional - only needed when running with LiveKit)
try:
    from livekit import rtc
    from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

    # Stubs for type hints when LiveKit not installed
    class VideoGenerator:
        pass

    class AudioSegmentEnd:
        pass

    rtc = None

from .adaptive_chunker import AdaptiveChunkConfig
from .adaptive_producer import AdaptiveFrameGenerator
from .events import FrameType, OutputFrame
from .frame_queue import FrameQueue
from .idle_injector import IdleFramePusher

# Lazy import model components to avoid loading at import time
_InferenceAgent = None
_AppConfig = None
_DefaultAgent = None


def _get_model_classes():
    """Lazy load model classes to avoid loading at import time."""
    global _InferenceAgent, _AppConfig, _DefaultAgent
    if _InferenceAgent is None:
        import importlib
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "avatar_models", "imtalker"))
        module = importlib.import_module("app")
        _InferenceAgent = module.InferenceAgent
        _AppConfig = module.AppConfig
        _DefaultAgent = getattr(module, "agent", None)
    return _InferenceAgent, _AppConfig


def preload_model(device: str = "cuda"):
    """
    Preload the IMTalker model for use in prewarm.

    Call this in prewarm() and pass the result to TalkBoxGenerator.

    Returns:
        InferenceAgent instance
    """
    print("TalkBox: Preloading model...")
    InferenceAgent, AppConfig = _get_model_classes()
    if _DefaultAgent is not None and str(_DefaultAgent.device) == device:
        print("TalkBox: Reusing model initialized by IMTalker.")
        return _DefaultAgent
    opt = AppConfig()
    opt.device = device
    agent = InferenceAgent(opt)
    print("TalkBox: Model preloaded.")
    return agent


@dataclass
class TalkBoxOptions:
    """Configuration options for TalkBox generator."""

    video_width: int = 512
    video_height: int = 512
    video_fps: int = 25
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # Chunk configuration
    min_chunk_frames: int = 10  # 400ms
    max_chunk_frames: int = 50  # 2000ms
    default_chunk_frames: int = 25  # 1000ms


class TalkBoxGenerator(VideoGenerator):
    """
    LiveKit VideoGenerator implementation for TalkBox.

    Wraps the TalkBox frame generation pipeline and exposes it through
    the LiveKit VideoGenerator interface for real-time WebRTC streaming.

    Usage:
        generator = TalkBoxGenerator(
            source_image_path="avatar.png",
            idle_cache_path="outputs/cache/imtalker_idle.pt",
        )
        await generator.start()

        # Used by AvatarRunner
        runner = AvatarRunner(room, video_gen=generator, ...)
    """

    def __init__(
        self,
        source_image_path: str,
        idle_cache_path: str,
        options: TalkBoxOptions | None = None,
        device: str = "cuda",
        preloaded_agent=None,  # Pass pre-loaded InferenceAgent from prewarm
    ):
        if not LIVEKIT_AVAILABLE:
            raise ImportError(
                "LiveKit SDK not installed. Install with: pip install livekit livekit-agents"
            )

        self._options = options or TalkBoxOptions()
        self._device = device
        self._source_image_path = source_image_path
        self._idle_cache_path = idle_cache_path
        self._preloaded_agent = preloaded_agent

        # State
        self._started = False
        self._stop_event = asyncio.Event()
        self._speaking = False

        # Audio resampler (initialized on first audio frame)
        self._resampler: rtc.AudioResampler | None = None

        # Components (initialized in start())
        self._agent = None
        self._frame_queue: FrameQueue | None = None
        self._audio_queue: queue.Queue | None = None
        self._producer: AdaptiveFrameGenerator | None = None
        self._idle_pusher: IdleFramePusher | None = None
        self._inference_lock = threading.RLock()

    async def start(self):
        """Initialize and start the generator."""
        if self._started:
            return

        print("TalkBoxGenerator: Initializing...")

        # Run initialization in thread pool (blocking operations)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._initialize_sync)

        self._started = True
        print("TalkBoxGenerator: Ready.")

    def _initialize_sync(self):
        """Synchronous initialization (runs in thread pool)."""
        InferenceAgent, AppConfig = _get_model_classes()

        # 1. Load model (or use preloaded)
        if self._preloaded_agent is not None:
            print("TalkBoxGenerator: Using preloaded agent")
            self._agent = self._preloaded_agent
            opt = AppConfig()
            opt.device = self._device
        else:
            print("TalkBoxGenerator: Loading model (no preloaded agent)")
            opt = AppConfig()
            opt.device = self._device
            self._agent = InferenceAgent(opt)

        # 2. Pre-process source identity
        img_pil = Image.open(self._source_image_path).convert("RGB")
        s_pil = self._agent.data_processor.process_img(img_pil)
        s_tensor = self._agent.data_processor.transform(s_pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            f_r, g_r = self._agent.renderer.dense_feature_encoder(s_tensor)
            t_lat = self._agent.renderer.latent_token_encoder(s_tensor)
            if isinstance(t_lat, tuple):
                t_lat = t_lat[0]
            ta_r = self._agent.renderer.adapt(t_lat, g_r)
            m_r = self._agent.renderer.latent_token_decoder(ta_r)

        source_features = {"f_r": f_r, "g_r": g_r, "t_lat": t_lat, "m_r": m_r}

        # 3. Create components
        self._frame_queue = FrameQueue(max_size=200, history_size=20)
        self._audio_queue = queue.Queue()

        # Chunk config
        chunk_config = AdaptiveChunkConfig(
            min_chunk_frames=self._options.min_chunk_frames,
            max_chunk_frames=self._options.max_chunk_frames,
            default_chunk_frames=self._options.default_chunk_frames,
        )

        # Idle pusher
        self._idle_pusher = IdleFramePusher(
            frame_queue=self._frame_queue,
            idle_cache_path=self._idle_cache_path,
            fps=self._options.video_fps,
            agent=self._agent,
            source_features=source_features,
            inference_lock=self._inference_lock,
        )

        # Producer
        self._producer = AdaptiveFrameGenerator(
            agent=self._agent,
            frame_queue=self._frame_queue,
            source_features=source_features,
            input_audio_queue=self._audio_queue,
            opt=opt,
            idle_pusher=self._idle_pusher,
            config=chunk_config,
            inference_lock=self._inference_lock,
        )

        # Start threads
        print("TalkBoxGenerator: Starting IdlePusher thread...")
        self._idle_pusher.start()
        print("TalkBoxGenerator: Starting Producer thread...")
        self._producer.start()
        print("TalkBoxGenerator: Threads started, waiting for frames...")

    async def stop(self):
        """Stop the generator and cleanup."""
        if not self._started:
            return

        print("TalkBoxGenerator: Stopping...")
        self._stop_event.set()

        # Stop threads
        if self._producer:
            self._producer.stop_event.set()
            self._producer.join(timeout=2.0)

        if self._idle_pusher:
            self._idle_pusher.stop_event.set()
            self._idle_pusher.join(timeout=2.0)

        self._started = False
        print("TalkBoxGenerator: Stopped.")

    # --- VideoGenerator Interface ---

    async def push_audio(self, frame: "rtc.AudioFrame | AudioSegmentEnd") -> None:
        """
        Receive audio from TTS and push to the generation pipeline.

        Args:
            frame: Audio frame from LiveKit or AudioSegmentEnd marker
        """
        if isinstance(frame, AudioSegmentEnd):
            # Audio stream ended - signal to producer to flush remaining audio
            print("TalkBoxGenerator: AudioSegmentEnd received, signaling producer to flush")
            self._speaking = False

            if self._resampler is not None:
                for rf in self._resampler.flush():
                    self._producer.enqueue_audio(bytes(rf.data))

            # Tell producer to flush remaining audio and mark final chunk
            if self._producer:
                self._producer.end_audio_stream()
            self._resampler = None
            return

        # Debug: Log audio frame receipt
        audio_bytes = len(frame.data) if hasattr(frame, "data") else 0
        if not self._speaking:
            print(
                f"TalkBoxGenerator: First audio frame received! sample_rate={frame.sample_rate}, bytes={audio_bytes}"
            )

        # Resample if needed (LiveKit often sends 24kHz or 48kHz)
        if frame.sample_rate != self._options.audio_sample_rate:
            if self._resampler is None:
                print(
                    f"TalkBoxGenerator: Creating resampler {frame.sample_rate}Hz -> {self._options.audio_sample_rate}Hz"
                )
                self._resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=self._options.audio_sample_rate,
                    num_channels=self._options.audio_channels,
                )

            # Resample
            resampled_frames = self._resampler.push(frame)
            for rf in resampled_frames:
                self._producer.enqueue_audio(bytes(rf.data))
        else:
            # Direct push
            self._producer.enqueue_audio(bytes(frame.data))

        # Signal speaking state (but DON'T stop idle pusher yet - wait for frames to be ready)
        if not self._speaking:
            self._speaking = True
            print(
                f"TalkBoxGenerator: Speaking state activated, audio queue size: {self._audio_queue.qsize()}"
            )
            # NOTE: We no longer call set_producer_active(True) here.
            # The producer will take over naturally when speaking frames arrive,
            # and idle pusher will stop filling once queue is full.
            # This prevents a gap in frame delivery during batch processing.

    def clear_buffer(self) -> None:
        """
        Handle interruption (barge-in).

        Called by AvatarRunner when user interrupts the avatar.
        """
        print("TalkBoxGenerator: Interrupt signal received")

        if not self._started:
            return

        # Get last state before reset
        published = self._frame_queue.peek_history()
        last_latent = (
            published.motion_latent
            if published is not None
            else self._producer.last_generated_latent
        )
        last_frame = (
            published.video_frame if published is not None else self._producer.last_generated_frame
        )
        was_speaking = bool(
            (published is not None and published.type != FrameType.IDLE)
            or self._producer.is_speaking
            or self._speaking
        )

        # Reset producer
        self._producer.hard_reset()

        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Purge frame queue
        self._frame_queue.purge()

        # Trigger smooth transition back to idle
        if last_latent is not None and was_speaking:
            self._idle_pusher.transition_from_latent(
                last_latent,
                last_frame,
                purge_before=True,
            )
        else:
            with self._idle_pusher.transition_lock, self._inference_lock:
                self._frame_queue.purge()
                self._idle_pusher.set_producer_active(False)

        self._speaking = False

        # Flush and discard any in-flight resampler state after interruption.
        if self._resampler:
            self._resampler.flush()
            self._resampler = None

    async def __aiter__(self) -> AsyncIterator["rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd"]:
        """
        Yield video and audio frames continuously.

        This is the main output loop that AvatarRunner consumes.
        Yields rtc.VideoFrame and rtc.AudioFrame for WebRTC streaming.

        Frames are paced here before yielding to avoid bursty delivery into LiveKit.
        """
        frame_count = 0
        audio_frame_count = 0
        video_frame_count = 0
        in_speech_segment = False  # Track if we're in a speech segment

        # Timing tracking for burst detection
        last_yield_time = None
        burst_frames = []  # Track inter-frame times during speech

        # Frame pacing to output at steady 25fps
        frame_interval = 1.0 / self._options.video_fps  # 40ms for 25fps
        next_frame_time = None  # Will be set on first frame

        print("TalkBoxGenerator: Starting frame iterator")

        while not self._stop_event.is_set():
            try:
                # Pull frame from queue (blocking with timeout)
                frame: OutputFrame | None = self._frame_queue.get(timeout=0.05)

                if frame is None:
                    # Buffer empty - yield to event loop briefly
                    await asyncio.sleep(0.01)
                    continue

                frame_count += 1
                current_time = time.time()
                queue_size = self._frame_queue.qsize()

                # Calculate time since last frame
                if last_yield_time is not None:
                    delta_ms = (current_time - last_yield_time) * 1000
                else:
                    delta_ms = 0

                # Detailed logging for SPEAKING frames to detect bursts
                if frame.type == FrameType.SPEAKING:
                    burst_frames.append(delta_ms)
                elif len(burst_frames) > 0:
                    # Just finished a burst of speaking frames - analyze
                    avg_delta = sum(burst_frames) / len(burst_frames) if burst_frames else 0
                    expected_delta = 1000 / self._options.video_fps  # 40ms for 25fps
                    print(
                        f"BURST_ANALYSIS: {len(burst_frames)} speaking frames, avg_delta={avg_delta:.2f}ms (expected={expected_delta:.1f}ms)"
                    )
                    burst_frames = []

                if frame_count % 50 == 1:
                    print(
                        f"TalkBoxGenerator: Yielding frame {frame_count}, queue size: {queue_size}, type: {frame.type}"
                    )

                last_yield_time = current_time

                # Convert to rtc.VideoFrame
                video_frame = rtc.VideoFrame(
                    width=self._options.video_width,
                    height=self._options.video_height,
                    type=rtc.VideoBufferType.RGB24,
                    data=frame.video_frame.tobytes(),
                )

                # Yield video frame
                yield video_frame
                video_frame_count += 1

                # Yield audio frame if present
                if frame.audio_frame is not None and len(frame.audio_frame) > 0:
                    audio_bytes_len = len(frame.audio_frame)
                    audio_frame = rtc.AudioFrame(
                        data=frame.audio_frame,
                        sample_rate=self._options.audio_sample_rate,
                        num_channels=self._options.audio_channels,
                        samples_per_channel=audio_bytes_len // 2,
                    )
                    yield audio_frame
                    audio_frame_count += 1

                # Log stats periodically
                if frame_count % 50 == 1:
                    print(
                        f"TalkBoxGenerator: frame={frame_count}, video={video_frame_count}, audio={audio_frame_count}, type={frame.type}, queue={self._frame_queue.qsize()}"
                    )

                # Track speech segment state
                if frame.type == FrameType.SPEAKING:
                    in_speech_segment = True

                # Yield AudioSegmentEnd based on final_chunk flag OR transition to idle
                # Use final_chunk if available, otherwise detect transition
                if frame.final_chunk:
                    print("TalkBoxGenerator: final_chunk=True, yielding AudioSegmentEnd")
                    yield AudioSegmentEnd()
                    in_speech_segment = False
                elif in_speech_segment and frame.type == FrameType.IDLE:
                    # Fallback: transition from speaking to idle without final_chunk
                    print("TalkBoxGenerator: Speech->Idle transition, yielding AudioSegmentEnd")
                    yield AudioSegmentEnd()
                    in_speech_segment = False

                # Pace output at target FPS using wall-clock time
                # This prevents bursting frames to AVSynchronizer
                now = time.time()
                if next_frame_time is None:
                    next_frame_time = now + frame_interval
                else:
                    sleep_time = next_frame_time - now
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    next_frame_time += frame_interval
                    # Reset if we fell too far behind (> 200ms)
                    if next_frame_time < now - 0.2:
                        print(
                            f"PACING: Fell behind by {(now - next_frame_time) * 1000:.1f}ms, resetting timer"
                        )
                        next_frame_time = now + frame_interval

                # Log pacing every 25 frames (1 second) - use delta_ms which measures actual interval
                if frame_count % 25 == 0 and frame_count > 0:
                    print(
                        f"PACING: frame={frame_count}, actual_interval={delta_ms:.1f}ms (target=40ms), type={frame.type}"
                    )

            except queue.Empty:
                if self._stop_event.is_set():
                    print("TalkBoxGenerator: Stop event set, exiting iterator")
                    break
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"TalkBoxGenerator: Iterator error: {e}")
                raise

        print("TalkBoxGenerator: Frame iterator stopped")

    # --- Properties ---

    @property
    def video_width(self) -> int:
        return self._options.video_width

    @property
    def video_height(self) -> int:
        return self._options.video_height

    @property
    def video_fps(self) -> int:
        return self._options.video_fps

    @property
    def audio_sample_rate(self) -> int:
        return self._options.audio_sample_rate

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def get_output_frame_nowait(self) -> OutputFrame:
        """Return the next generated frame without blocking the asyncio loop."""
        if self._frame_queue is None:
            raise RuntimeError("TalkBoxGenerator is not started")
        return self._frame_queue.get_nowait()

    def get_idle_fallback(self) -> OutputFrame:
        """Advance cached idle motion while the CUDA producer is busy."""
        if self._idle_pusher is None or self._frame_queue is None:
            raise RuntimeError("TalkBoxGenerator is not started")
        frame = self._idle_pusher.next_idle_frame()
        self._frame_queue.remember(frame)
        return frame
