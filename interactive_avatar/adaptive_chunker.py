"""
Adaptive audio chunking for low-latency frame generation.

Implements queue-aware deadline tracking for adaptive generation:
- First chunk: Emit as soon as minimum audio arrives
- Subsequent chunks: Emit when output queue drops below threshold
- Overlapping context: Carry motion latents between chunks for smooth transitions
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class AdaptiveChunkConfig:
    """Configuration for adaptive audio chunking."""

    # Frame timing
    fps: int = 25
    sample_rate: int = 16000
    bytes_per_sample: int = 2  # 16-bit audio

    # Chunk size bounds (in frames)
    min_chunk_frames: int = 10  # ~400ms - minimum for first chunk
    max_chunk_frames: int = 50  # ~2000ms - maximum chunk size
    default_chunk_frames: int = 25  # ~1000ms - default for steady state

    # Queue management
    queue_low_threshold: int = 8  # Emit when queue drops below this
    queue_high_threshold: int = 20  # Don't emit if queue is above this

    # Context overlap (in frames)
    num_prev_frames: int = 10  # Motion latent history for continuity

    @property
    def bytes_per_frame(self) -> int:
        """Audio bytes per video frame."""
        return int(self.sample_rate * self.bytes_per_sample / self.fps)

    @property
    def min_chunk_bytes(self) -> int:
        """Minimum audio bytes before first chunk can emit."""
        return self.min_chunk_frames * self.bytes_per_frame

    @property
    def max_chunk_bytes(self) -> int:
        """Maximum audio bytes per chunk."""
        return self.max_chunk_frames * self.bytes_per_frame

    @property
    def default_chunk_bytes(self) -> int:
        """Default chunk size in bytes."""
        return self.default_chunk_frames * self.bytes_per_frame


class QueueSizeProvider(Protocol):
    """Protocol for getting current queue size."""

    def qsize(self) -> int: ...


@dataclass
class ChunkDecision:
    """Result of chunk emission decision."""

    should_emit: bool
    chunk_size_frames: int
    reason: str


class AdaptiveDeadlineTracker:
    """
    Queue-aware deadline tracker for adaptive chunk emission.

    Monitors the output frame queue to decide when to emit chunks:
    - First chunk: Emit immediately when minimum audio is available
    - Subsequent: Emit when queue drops below threshold
    - Never emit if queue is too full (backpressure)
    """

    def __init__(
        self,
        config: AdaptiveChunkConfig,
        output_queue: QueueSizeProvider,
    ):
        self.config = config
        self.output_queue = output_queue
        self._first_chunk = True
        self._last_emit_time: float | None = None
        self._lock = threading.Lock()

    def reset(self):
        """Reset tracker state for new audio stream."""
        with self._lock:
            self._first_chunk = True
            self._last_emit_time = None

    def should_emit_chunk(self, available_audio_bytes: int, is_stream_done: bool) -> ChunkDecision:
        """
        Decide whether to emit a chunk based on queue state and available audio.

        Args:
            available_audio_bytes: Bytes of audio currently buffered
            is_stream_done: True if audio stream has ended

        Returns:
            ChunkDecision with emit flag, recommended chunk size, and reason
        """
        with self._lock:
            queue_size = self.output_queue.qsize()
            available_frames = available_audio_bytes // self.config.bytes_per_frame

            # Case 1: Stream is done - flush remaining audio
            if is_stream_done and available_audio_bytes > 0:
                chunk_frames = min(
                    math.ceil(available_audio_bytes / self.config.bytes_per_frame),
                    self.config.max_chunk_frames,
                )
                if chunk_frames > 0:
                    return ChunkDecision(
                        should_emit=True,
                        chunk_size_frames=chunk_frames,
                        reason=f"stream_done (flush {chunk_frames} frames)",
                    )
                return ChunkDecision(False, 0, "stream_done_empty")

            # Case 2: Not enough audio for minimum chunk
            if available_audio_bytes < self.config.min_chunk_bytes:
                return ChunkDecision(
                    should_emit=False,
                    chunk_size_frames=0,
                    reason=f"insufficient_audio ({available_frames}/{self.config.min_chunk_frames} frames)",
                )

            # Case 3: Queue is too full - apply backpressure
            if queue_size > self.config.queue_high_threshold:
                return ChunkDecision(
                    should_emit=False,
                    chunk_size_frames=0,
                    reason=f"backpressure (queue={queue_size})",
                )

            # Case 4: First chunk - emit immediately with minimum size
            if self._first_chunk:
                self._first_chunk = False
                self._last_emit_time = time.time()
                chunk_frames = min(available_frames, self.config.min_chunk_frames)
                return ChunkDecision(
                    should_emit=True,
                    chunk_size_frames=chunk_frames,
                    reason=f"first_chunk ({chunk_frames} frames)",
                )

            # Case 5: Queue is getting low - emit to prevent starvation
            if queue_size < self.config.queue_low_threshold:
                # Adaptive chunk size based on how empty the queue is
                urgency = 1.0 - (queue_size / self.config.queue_low_threshold)
                target_frames = int(
                    self.config.min_chunk_frames
                    + urgency * (self.config.default_chunk_frames - self.config.min_chunk_frames)
                )
                chunk_frames = min(available_frames, target_frames, self.config.max_chunk_frames)
                self._last_emit_time = time.time()
                return ChunkDecision(
                    should_emit=True,
                    chunk_size_frames=chunk_frames,
                    reason=f"queue_low (queue={queue_size}, emit {chunk_frames} frames)",
                )

            # Case 6: Have enough for a full default chunk
            if available_frames >= self.config.default_chunk_frames:
                self._last_emit_time = time.time()
                return ChunkDecision(
                    should_emit=True,
                    chunk_size_frames=self.config.default_chunk_frames,
                    reason=f"buffer_full ({available_frames} frames available)",
                )

            # Case 7: Wait for more audio
            return ChunkDecision(
                should_emit=False,
                chunk_size_frames=0,
                reason=f"waiting (queue={queue_size}, audio={available_frames} frames)",
            )


@dataclass
class AudioChunk:
    """Represents a chunk of audio ready for processing."""

    audio_bytes: bytes
    num_frames: int
    is_first_chunk: bool
    is_final_chunk: bool
    chunk_index: int


class AdaptiveAudioBuffer:
    """
    Buffers incoming audio and emits chunks adaptively based on queue state.

    Features:
    - Accumulates audio from streaming input
    - Uses deadline tracker to decide when to emit
    - Tracks chunk boundaries for context management
    """

    def __init__(
        self,
        config: AdaptiveChunkConfig,
        deadline_tracker: AdaptiveDeadlineTracker,
    ):
        self.config = config
        self.deadline_tracker = deadline_tracker
        self._buffer = bytearray()
        self._is_done = False
        self._chunk_index = 0
        self._is_first = True
        self._completion_emitted = False
        self._lock = threading.Lock()

    def push_audio(self, audio_bytes: bytes):
        """Add audio bytes to the buffer."""
        with self._lock:
            self._buffer.extend(audio_bytes)

    def mark_done(self):
        """Mark the audio stream as complete."""
        with self._lock:
            self._is_done = True

    def reset(self):
        """Reset buffer for new audio stream."""
        with self._lock:
            self._buffer.clear()
            self._is_done = False
            self._chunk_index = 0
            self._is_first = True
            self._completion_emitted = False
            self.deadline_tracker.reset()

    def try_get_chunk(self) -> AudioChunk | None:
        """
        Try to get the next audio chunk if conditions are met.

        Returns:
            AudioChunk if ready to emit, None otherwise
        """
        with self._lock:
            if self._is_done and not self._buffer and not self._completion_emitted:
                self._completion_emitted = True
                return AudioChunk(
                    audio_bytes=b"",
                    num_frames=0,
                    is_first_chunk=self._is_first,
                    is_final_chunk=True,
                    chunk_index=self._chunk_index,
                )

            decision = self.deadline_tracker.should_emit_chunk(
                available_audio_bytes=len(self._buffer),
                is_stream_done=self._is_done,
            )

            if not decision.should_emit:
                return None

            # Extract chunk from buffer
            chunk_bytes = decision.chunk_size_frames * self.config.bytes_per_frame
            audio_data = bytes(self._buffer[:chunk_bytes]).ljust(chunk_bytes, b"\0")
            self._buffer = self._buffer[chunk_bytes:]

            # Check if this is final chunk
            is_final = self._is_done and len(self._buffer) < self.config.bytes_per_frame
            if is_final:
                self._buffer.clear()
                self._completion_emitted = True

            chunk = AudioChunk(
                audio_bytes=audio_data,
                num_frames=decision.chunk_size_frames,
                is_first_chunk=self._is_first,
                is_final_chunk=is_final,
                chunk_index=self._chunk_index,
            )

            self._is_first = False
            self._chunk_index += 1

            return chunk

    @property
    def buffered_frames(self) -> int:
        """Number of complete frames currently buffered."""
        with self._lock:
            return len(self._buffer) // self.config.bytes_per_frame

    @property
    def is_done(self) -> bool:
        """Whether audio stream has ended."""
        with self._lock:
            return self._is_done
