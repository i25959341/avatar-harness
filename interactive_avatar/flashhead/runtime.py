from __future__ import annotations

import sys
import threading
import time
from collections import deque

import numpy as np
import torch

from .config import FlashHeadConfig


class FlashHeadRuntime:
    """Own one mutable FlashHead pipeline and its rolling audio/motion state."""

    def __init__(self, config: FlashHeadConfig) -> None:
        self.config = config
        self.pipeline = None
        self._lock = threading.Lock()
        self._audio_context: deque[float] = deque()
        self._loaded = False

    def load(self) -> float:
        if self._loaded:
            return 0.0
        self.config.validate()
        sys.path.insert(0, str(self.config.flashhead_dir))
        self._install_protobuf_compatibility()
        from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline

        started = time.perf_counter()
        self.pipeline = FlashHeadPipeline(
            checkpoint_dir=str(self.config.checkpoint_dir),
            model_type="lite",
            wav2vec_dir=str(self.config.wav2vec_dir),
            device="cuda",
        )
        self.pipeline.prepare_params(
            cond_image_path_or_dir=str(self.config.source_image),
            target_size=(self.config.height, self.config.width),
            frame_num=self.config.chunk_frames + self.config.motion_frames,
            motion_frames_num=self.config.motion_frames,
            sampling_steps=4,
            seed=self.config.seed,
            shift=5.0,
            color_correction_strength=1.0,
            use_face_crop=False,
        )
        self._reset_audio_context()
        self._loaded = True
        return time.perf_counter() - started

    def warmup(self) -> float:
        self._require_loaded()
        started = time.perf_counter()
        self.generate_chunk(bytes(self.config.bytes_per_chunk))
        assert self.pipeline is not None
        self.pipeline.reset_person_name()
        self._reset_audio_context()
        return time.perf_counter() - started

    def generate_chunk(self, pcm_s16le: bytes) -> np.ndarray:
        with self._lock:
            return self._generate_chunk_locked(pcm_s16le)

    def generate_from_motion(
        self, motion_frames_rgb: list[np.ndarray], pcm_s16le: bytes
    ) -> np.ndarray:
        with self._lock:
            self._set_motion_locked(motion_frames_rgb)
            self._reset_audio_context()
            return self._generate_chunk_locked(pcm_s16le)

    def generate_vae_transition(
        self,
        source_frames_rgb: list[np.ndarray],
        target_frames_rgb: list[np.ndarray],
        transition_frames: int,
    ) -> np.ndarray:
        """Interpolate published motion into an idle window in VAE latent space."""
        if transition_frames < 2:
            raise ValueError("transition_frames must be at least 2")
        with self._lock:
            self._require_loaded()
            assert self.pipeline is not None
            source = self._frames_to_tensor(self._motion_window(source_frames_rgb))
            target = self._frames_to_tensor(self._motion_window(target_frames_rgb))
            source_latent = self.pipeline.vae.encode(source)
            target_latent = self.pipeline.vae.encode(target)

            bridge: list[np.ndarray] = []
            for step in range(transition_frames):
                progress = (step + 1) / (transition_frames + 1)
                alpha = progress * progress * (3.0 - 2.0 * progress)
                latent = torch.lerp(source_latent, target_latent, alpha)
                decoded = self.pipeline.vae.decode(latent)
                frame = decoded[0, :, -1].float().permute(1, 2, 0)
                bridge.append(
                    ((frame + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
                )
            return np.stack(bridge)

    def _generate_chunk_locked(self, pcm_s16le: bytes) -> np.ndarray:
        self._require_loaded()
        if len(pcm_s16le) != self.config.bytes_per_chunk:
            raise ValueError(
                f"expected {self.config.bytes_per_chunk} PCM bytes, got {len(pcm_s16le)}"
            )
        assert self.pipeline is not None
        samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_context.extend(samples.tolist())
        audio = np.asarray(self._audio_context, dtype=np.float32)
        embedding = self.pipeline.preprocess_audio(
            audio,
            sr=self.config.sample_rate,
            fps=self.config.fps,
        )
        audio_end = self.config.audio_context_seconds * self.config.fps
        audio_start = audio_end - (self.config.chunk_frames + self.config.motion_frames)
        indices = torch.arange(-2, 3)
        centers = torch.arange(audio_start, audio_end).unsqueeze(1) + indices.unsqueeze(0)
        centers = torch.clamp(centers, min=0, max=audio_end - 1)
        conditioned = embedding[centers][None, ...].contiguous().to(self.pipeline.device)

        generated = self.pipeline.generate(conditioned)
        frames = (
            ((generated + 1.0) / 2.0).permute(1, 2, 3, 0).clip(0, 1).mul(255).byte().cpu().numpy()
        )
        return frames[self.config.motion_frames :]

    def _set_motion_locked(self, frames_rgb: list[np.ndarray]) -> None:
        self._require_loaded()
        tensor = self._frames_to_tensor(self._motion_window(frames_rgb))
        assert self.pipeline is not None
        self.pipeline.latent_motion_frames = self.pipeline.vae.encode(tensor)

    def _motion_window(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        if not frames_rgb:
            raise ValueError("at least one motion frame is required")
        frames = list(frames_rgb[-self.config.motion_frames :])
        while len(frames) < self.config.motion_frames:
            frames.insert(0, frames[0])
        return frames

    @staticmethod
    def _frames_to_tensor(frames_rgb: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(frames_rgb)
        tensor = (
            torch.from_numpy(array)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device="cuda", dtype=torch.bfloat16)
        )
        return (tensor / 127.5) - 1.0

    def _reset_audio_context(self) -> None:
        length = self.config.audio_context_seconds * self.config.sample_rate
        self._audio_context = deque([0.0] * length, maxlen=length)

    def _require_loaded(self) -> None:
        if not self._loaded or self.pipeline is None:
            raise RuntimeError("FlashHead runtime is not loaded")

    @staticmethod
    def _install_protobuf_compatibility() -> None:
        # MediaPipe still calls this API while LiveKit requires Protobuf 5+.
        from google.protobuf import message_factory

        if not hasattr(message_factory.MessageFactory, "GetPrototype"):

            def get_prototype(self: object, descriptor: object) -> object:
                del self
                return message_factory.GetMessageClass(descriptor)

            message_factory.MessageFactory.GetPrototype = get_prototype
