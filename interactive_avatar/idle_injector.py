from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import torch

from .events import FrameType, OutputFrame
from .frame_queue import FrameQueue


class IdleFramePusher(threading.Thread):
    def __init__(
        self,
        frame_queue: FrameQueue,
        idle_cache_path: str,
        fps: int = 25,
        threshold: int = 5,
        agent=None,
        source_features: dict = None,
        inference_lock: threading.RLock | None = None,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.fps = fps
        self.threshold = threshold  # Inject if queue size < threshold
        self.stop_event = threading.Event()
        self.inference_lock = inference_lock or threading.RLock()

        # Renderer components for latent interpolation
        self.agent = agent
        self.source_features = source_features
        if source_features:
            self.t_lat = source_features["t_lat"]
            self.m_r = source_features["m_r"]
            self.g_r = source_features["g_r"]
            self.f_r = source_features["f_r"]
            self.device = agent.device if agent else "cuda"

        # Load Cache
        print(f"IdleFramePusher: Loading cache from {idle_cache_path}...")
        self.idle_frames: list[OutputFrame] = torch.load(idle_cache_path, weights_only=False)
        print(f"IdleFramePusher: Loaded {len(self.idle_frames)} frames.")

        self.index = 0
        self.index_lock = threading.Lock()
        self.producer_active = False  # Flag to signal if main producer is working
        self.transition_lock = threading.Lock()  # Prevent idle frames during transition

        # Pre-stack all motion latents for fast query [N, D]
        # output_frame.motion_latent is [1, D] or [D]
        self.latent_stack = torch.cat(
            [f.motion_latent.view(1, -1) for f in self.idle_frames], dim=0
        )
        # Normalize for better comparison if needed, but raw L2 is okay for first pass if scale is consistent.

    def set_producer_active(self, active: bool):
        self.producer_active = active

    def _smoothstep(self, t: float) -> float:
        """Ease-in-out interpolation for more natural motion."""
        return t * t * (3 - 2 * t)

    def _decode_latent(self, motion_latent: torch.Tensor) -> np.ndarray:
        """Decode a motion latent to RGB frame using the renderer."""
        with torch.no_grad():
            # Ensure latent is on correct device and shape [1, D]
            latent = motion_latent.view(1, -1).to(self.device)

            # Decode through renderer (same as producer.py)
            ta_c = self.agent.renderer.adapt(latent, self.g_r)
            m_c = self.agent.renderer.latent_token_decoder(ta_c)
            out_frame = self.agent.renderer.decode(m_c, self.m_r, self.f_r)

            # Convert to numpy [H, W, 3] uint8
            frame_np = out_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
            return frame_np

    def next_idle_frame(self) -> OutputFrame:
        """Return the next cached idle frame for a real-time fallback publisher."""
        with self.index_lock:
            frame = self.idle_frames[self.index]
            self.index = (self.index + 1) % len(self.idle_frames)
            return frame

    def transition_from_latent(
        self,
        target_latent: torch.Tensor,
        last_frame_pixels: np.ndarray = None,
        *,
        mark_final: bool = False,
        purge_before: bool = False,
    ):
        """
        Smooth transition from speaking to idle using motion latent interpolation.
        target_latent: [1, D] or [D] tensor from the last speaking frame.
        last_frame_pixels: [H, W, 3] numpy array (RGB) - unused now but kept for API compatibility.
        """
        # Acquire lock to prevent idle pusher from interfering
        with self.transition_lock:
            self._do_transition(
                target_latent,
                last_frame_pixels,
                mark_final=mark_final,
                purge_before=purge_before,
            )

    def _do_transition(
        self,
        target_latent: torch.Tensor,
        last_frame_pixels: np.ndarray = None,
        *,
        mark_final: bool = False,
        purge_before: bool = False,
    ):
        """Internal transition logic, called with lock held."""
        with self.inference_lock:
            self._render_transition(
                target_latent,
                last_frame_pixels,
                mark_final=mark_final,
                purge_before=purge_before,
            )

    def _render_transition(
        self,
        target_latent: torch.Tensor,
        last_frame_pixels: np.ndarray = None,
        *,
        mark_final: bool = False,
        purge_before: bool = False,
    ):
        """Render a transition while holding the shared model lock."""
        if purge_before:
            self.frame_queue.purge()
        speaking_latent = target_latent.view(1, -1).cpu()

        # L2 Distance with all cached latents to find best idle frame
        dists = torch.norm(self.latent_stack - speaking_latent, dim=1)
        best_idx = int(torch.argmin(dists).item())

        print(
            f"IdleFramePusher: Latent Interpolation! Target idle frame {best_idx} (Dist: {dists[best_idx]:.4f})"
        )

        # Get the target idle latent (also on CPU)
        idle_latent = self.idle_frames[best_idx].motion_latent.view(1, -1).cpu()

        # Check if we have renderer access for proper interpolation
        if self.agent is not None and self.source_features is not None:
            steps = 8  # Number of interpolation frames
            silent_audio = self.idle_frames[best_idx].audio_frame

            for i in range(1, steps + 1):
                # Smoothstep interpolation for natural motion
                # t goes from 1/steps to 1.0 (e.g., 0.125, 0.25, ..., 1.0)
                # This ensures the last frame matches the idle target exactly
                t = i / steps
                t_smooth = self._smoothstep(t)

                # Interpolate in latent space
                z_interp = (1 - t_smooth) * speaking_latent + t_smooth * idle_latent
                z_interp = z_interp.to(self.device)

                # Decode to frame
                frame_np = self._decode_latent(z_interp)

                # Create transition frame
                transition_frame = OutputFrame(
                    video_frame=frame_np,
                    audio_frame=silent_audio,
                    motion_latent=z_interp.cpu(),
                    type=FrameType.TRANSITION,
                    final_chunk=mark_final and i == steps,
                )

                self.frame_queue.put(transition_frame)
        else:
            # Fallback to pixel blending if no renderer
            print("IdleFramePusher: No renderer available, falling back to pixel blend.")
            if last_frame_pixels is not None:
                target_frame = self.idle_frames[best_idx].video_frame
                if last_frame_pixels.shape == target_frame.shape:
                    steps = 5
                    silent_audio = self.idle_frames[best_idx].audio_frame
                    for i in range(1, steps + 1):
                        alpha = i / (steps + 1)
                        blended = cv2.addWeighted(
                            last_frame_pixels, 1.0 - alpha, target_frame, alpha, 0
                        )
                        interim_frame = OutputFrame(
                            video_frame=blended,
                            audio_frame=silent_audio,
                            motion_latent=target_latent,
                            type=FrameType.TRANSITION,
                            final_chunk=mark_final and i == steps,
                        )
                        self.frame_queue.put(interim_frame)

        self.index = best_idx
        self.producer_active = False  # Resume idle loop

    def run(self):
        print("IdleFramePusher: Thread started.")
        interval = 1.0 / (self.fps * 2)  # Check twice as fast as FPS

        while not self.stop_event.is_set():
            time.sleep(interval)

            # Logic:
            # 1. Main Producer corresponds to "Speaking".
            # 2. If Producer is Active, we DO NOT interfere (unless underflow protection needed, but let's stick to idle only for now).
            # 3. If Producer is Inactive (Idle), we keep the queue full.
            # 4. If a transition is in progress, wait for it to complete.

            if self.producer_active:
                continue

            # Don't push idle frames while transition is in progress
            if self.transition_lock.locked():
                continue

            current_size = self.frame_queue.qsize()
            if current_size < self.threshold:
                # Need to fill buffer
                needed = self.threshold - current_size

                # Push frames
                # We do this in a loop to keep it topped up
                # Note: This is a simple loop. For specialized loop points, we'd use the loop indices found earlier.
                # Here we just cycle the whole video.
                for _ in range(needed):
                    frame = self.next_idle_frame()
                    self.frame_queue.put(frame)

        print("IdleFramePusher: Thread stopped.")

    def stop(self):
        self.stop_event.set()
        self.join()
