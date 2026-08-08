"""
Adaptive frame generator with variable batch sizes for low-latency generation.

Key improvements over fixed-batch producer:
- First chunk emits with ~400ms audio (vs 2000ms)
- Subsequent chunks adapt based on output queue fullness
- Smooth context carryover between variable-sized chunks
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import torch

from .adaptive_chunker import (
    AdaptiveAudioBuffer,
    AdaptiveChunkConfig,
    AdaptiveDeadlineTracker,
    AudioChunk,
)
from .events import FrameType, OutputFrame
from .frame_queue import FrameQueue

if TYPE_CHECKING:
    from app import InferenceAgent


AUDIO_STREAM_END = object()


class AdaptiveFrameGenerator(threading.Thread):
    """
    Frame generator with adaptive chunk sizes for low-latency streaming.

    Uses queue-aware deadline tracking to decide when to generate frames,
    enabling much lower time-to-first-frame while maintaining quality.
    """

    def __init__(
        self,
        agent: InferenceAgent,
        frame_queue: FrameQueue,
        source_features: dict,
        input_audio_queue: queue.Queue,
        opt,
        idle_pusher=None,
        config: AdaptiveChunkConfig | None = None,
        inference_lock: threading.RLock | None = None,
    ):
        super().__init__()
        self.agent = agent
        self.frame_queue = frame_queue
        self.input_audio_queue = input_audio_queue
        self.opt = opt
        self.idle_pusher = idle_pusher
        self.stop_event = threading.Event()
        self.inference_lock = inference_lock or threading.RLock()

        # Configuration
        self.config = config or AdaptiveChunkConfig()

        # Adaptive chunking components
        self.deadline_tracker = AdaptiveDeadlineTracker(
            config=self.config,
            output_queue=self.frame_queue,
        )
        self.audio_buffer = AdaptiveAudioBuffer(
            config=self.config,
            deadline_tracker=self.deadline_tracker,
        )

        # Source features
        self.t_lat = source_features["t_lat"]
        self.m_r = source_features["m_r"]
        self.g_r = source_features["g_r"]
        self.f_r = source_features["f_r"]
        self.device = agent.device

        # State management
        self.state = None
        self.is_speaking = False
        self.last_generated_frame = None
        self.last_generated_latent = None
        self.interrupted = False
        self.generation_epoch = 0

        # Dimension info
        self.dim_c = getattr(opt, "dim_c", 64)
        self.dim_w = getattr(opt, "dim_motion", 128)

    def _init_state(self, batch_size=1):
        """Initialize flow matching state tensors."""
        num_prev = self.config.num_prev_frames
        return {
            "prev_x": torch.zeros(batch_size, num_prev, self.dim_w, device=self.device),
            "prev_a": torch.zeros(batch_size, num_prev, self.dim_c, device=self.device),
            "prev_gaze": torch.zeros(batch_size, num_prev, self.dim_c, device=self.device),
            "prev_pose": torch.zeros(batch_size, num_prev, self.dim_c, device=self.device),
            "prev_cam": torch.zeros(batch_size, num_prev, self.dim_c, device=self.device),
        }

    def run(self):
        print(
            f"AdaptiveFrameGenerator: Started (min={self.config.min_chunk_frames}, "
            f"max={self.config.max_chunk_frames} frames)"
        )

        if self.state is None:
            self.state = self._init_state()

        from torchdiffeq import odeint

        while not self.stop_event.is_set():
            try:
                # 1. Try to get audio from input queue
                try:
                    queued = self.input_audio_queue.get(timeout=0.05)
                    if (
                        isinstance(queued, tuple)
                        and len(queued) == 2
                        and isinstance(queued[0], int)
                    ):
                        item_epoch, chunk = queued
                    else:
                        item_epoch, chunk = self.generation_epoch, queued
                    if item_epoch != self.generation_epoch:
                        continue
                    if chunk is AUDIO_STREAM_END:
                        self.audio_buffer.mark_done()
                    else:
                        self.interrupted = False
                        self.is_speaking = True
                        if self.idle_pusher:
                            self.idle_pusher.set_producer_active(True)
                        self.audio_buffer.push_audio(chunk)
                except queue.Empty:
                    # Check if we should mark stream as done
                    if self.is_speaking and self.input_audio_queue.empty():
                        # Small delay to ensure no more audio is coming
                        pass  # Will be handled by audio buffer state

                # 2. Try to get a chunk to process
                audio_chunk = self.audio_buffer.try_get_chunk()

                if audio_chunk is not None:
                    chunk_epoch = self.generation_epoch
                    # Process this chunk
                    if audio_chunk.num_frames:
                        self._process_chunk(audio_chunk, odeint, chunk_epoch)

                    if self.interrupted or chunk_epoch != self.generation_epoch:
                        continue

                    # Check if stream ended
                    if audio_chunk.is_final_chunk:
                        print(
                            "AdaptiveFrameGenerator: Final chunk processed. Triggering Return-to-Idle."
                        )
                        self.is_speaking = False

                        if self.idle_pusher and self.last_generated_latent is not None:
                            last_latent = self.state["prev_x"][:, -1, :]
                            self.idle_pusher.transition_from_latent(
                                last_latent,
                                self.last_generated_frame,
                                mark_final=audio_chunk.num_frames == 0,
                            )

                        # Reset for next stream
                        self.audio_buffer.reset()

            except Exception as e:
                print(f"AdaptiveFrameGenerator Error: {e}")
                import traceback

                traceback.print_exc()
                continue

        print("AdaptiveFrameGenerator: Stopped.")

    def _process_chunk(self, chunk: AudioChunk, ode_func, generation_epoch: int):
        """
        Process an audio chunk of variable size.

        Args:
            chunk: AudioChunk with audio data and metadata
            ode_func: ODE solver function (odeint)
        """
        num_frames = chunk.num_frames

        # Convert audio bytes to float
        audio_np = np.frombuffer(chunk.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        with self.inference_lock, torch.no_grad():
            # A. Audio Features (wav2vec2)
            a_features = self.agent.data_processor.wav2vec_preprocessor(
                audio_np, sampling_rate=16000, return_tensors="pt"
            ).input_values.to(self.device)

            # B. Audio Encoder - KEY: pass variable num_frames as seq_len
            a_geo = self.agent.generator.audio_encoder.inference(a_features, seq_len=num_frames)
            a_geo = self.agent.generator.audio_projection(a_geo)  # (1, num_frames, dim_c)

            # C. Flow Matching with variable sequence length
            B, T, D = 1, num_frames, self.dim_c

            # Control signals (zeros for now)
            gaze = torch.zeros(B, T, D, device=self.device)
            pose = torch.zeros(B, T, D, device=self.device)
            cam = torch.zeros(B, T, D, device=self.device)

            # Initial noise
            x0 = torch.randn(B, T, self.dim_w, device=self.device)

            # Get previous state for context
            prev_x_in = self.state["prev_x"]
            prev_a_in = self.state["prev_a"]
            prev_gaze_in = self.state["prev_gaze"]
            prev_pose_in = self.state["prev_pose"]
            prev_cam_in = self.state["prev_cam"]

            # Transition logic: Seed from idle if coming from idle state
            if chunk.is_first_chunk:
                history_frame = self.frame_queue.peek_history()
                if history_frame is not None and history_frame.type == FrameType.IDLE:
                    motion = history_frame.motion_latent.to(self.device)
                    if motion.dim() == 2:
                        motion = motion.unsqueeze(1)
                    # Fill history with idle pose
                    self.state["prev_x"] = motion.repeat(1, self.config.num_prev_frames, 1)
                    self.state["prev_a"] = torch.zeros_like(self.state["prev_a"])
                    prev_x_in = self.state["prev_x"]
                    prev_a_in = self.state["prev_a"]

                # Speaking frames should take over immediately once they are ready.
                # Any queued idle frames at this point only add avoidable latency.
                self.frame_queue.purge()

            # ODE solver closure
            def sample_ode(tt, zt):
                out = self.agent.generator.fmt.forward_with_cfg(
                    t=tt.unsqueeze(0),
                    x=zt,
                    a=a_geo,
                    prev_x=prev_x_in,
                    prev_a=prev_a_in,
                    ref_x=self.t_lat,
                    gaze=gaze,
                    prev_gaze=prev_gaze_in,
                    pose=pose,
                    prev_pose=prev_pose_in,
                    cam=cam,
                    prev_cam=prev_cam_in,
                    a_cfg_scale=3.0,
                )
                # Output includes prev frames, slice to get current only
                return out[:, self.config.num_prev_frames :]

            # Solve ODE
            time_steps = torch.linspace(0, 1, 10, device=self.device)
            trajectory = ode_func(sample_ode, x0, time_steps, **self.agent.generator.odeint_kwargs)
            sample = trajectory[-1]  # (1, num_frames, dim_w)

            # D. Update state for next chunk (context carryover)
            # Use last num_prev_frames from this chunk for next chunk's context
            num_prev = self.config.num_prev_frames
            if num_frames >= num_prev:
                self.state["prev_x"] = sample[:, -num_prev:, :]
                self.state["prev_a"] = a_geo[:, -num_prev:, :]
                self.state["prev_gaze"] = gaze[:, -num_prev:, :]
                self.state["prev_pose"] = pose[:, -num_prev:, :]
                self.state["prev_cam"] = cam[:, -num_prev:, :]
            else:
                # Chunk smaller than context window - slide and append
                keep = num_prev - num_frames
                self.state["prev_x"] = torch.cat(
                    [self.state["prev_x"][:, -keep:, :], sample], dim=1
                )
                self.state["prev_a"] = torch.cat([self.state["prev_a"][:, -keep:, :], a_geo], dim=1)
                # Similar for gaze, pose, cam...
                self.state["prev_gaze"] = torch.cat(
                    [self.state["prev_gaze"][:, -keep:, :], gaze], dim=1
                )
                self.state["prev_pose"] = torch.cat(
                    [self.state["prev_pose"][:, -keep:, :], pose], dim=1
                )
                self.state["prev_cam"] = torch.cat(
                    [self.state["prev_cam"][:, -keep:, :], cam], dim=1
                )

            # E. Decode frames and push to queue
            bytes_per_frame = self.config.bytes_per_frame
            for t in range(num_frames):
                if self.interrupted or generation_epoch != self.generation_epoch:
                    print("AdaptiveFrameGenerator: Aborting chunk due to interrupt.")
                    break

                sample_t = sample[:, t, ...]  # (1, dim_w)

                # Decode to image
                ta_c = self.agent.renderer.adapt(sample_t, self.g_r)
                m_c = self.agent.renderer.latent_token_decoder(ta_c)
                out_frame = self.agent.renderer.decode(m_c, self.m_r, self.f_r)

                # Convert to numpy
                frame_np = out_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)

                # Get corresponding audio slice
                start_b = t * bytes_per_frame
                end_b = (t + 1) * bytes_per_frame
                audio_slice = chunk.audio_bytes[start_b:end_b]

                # Check if this is the last frame of the final chunk
                is_last_frame = (t == num_frames - 1) and chunk.is_final_chunk

                # Create output frame
                output_item = OutputFrame(
                    video_frame=frame_np,
                    audio_frame=audio_slice,
                    motion_latent=sample_t.clone(),
                    type=FrameType.SPEAKING,
                    final_chunk=is_last_frame,
                )

                # Cache for transitions
                self.last_generated_frame = frame_np
                self.last_generated_latent = sample_t.clone()

                # Push to output queue
                self.frame_queue.put(output_item)

                if is_last_frame:
                    print("AdaptiveFrameGenerator: Pushed final frame of speech segment")

    def push_audio(self, audio_bytes: bytes):
        """Push audio directly to the adaptive buffer (alternative to queue)."""
        self.audio_buffer.push_audio(audio_bytes)
        self.is_speaking = True
        self.interrupted = False

    def enqueue_audio(self, audio_bytes: bytes):
        """Queue PCM tagged with the current interruption epoch."""
        self.input_audio_queue.put((self.generation_epoch, audio_bytes))

    def end_audio_stream(self):
        """Signal that audio stream has ended."""
        self.input_audio_queue.put((self.generation_epoch, AUDIO_STREAM_END))

    def stop(self):
        """Stop the generator thread."""
        self.stop_event.set()
        self.join()

    def hard_reset(self):
        """Reset for interruption."""
        print("AdaptiveFrameGenerator: HARD RESET triggered.")
        self.generation_epoch += 1
        self.interrupted = True
        self.audio_buffer.reset()
        self.is_speaking = False
        if self.idle_pusher:
            self.idle_pusher.set_producer_active(False)
