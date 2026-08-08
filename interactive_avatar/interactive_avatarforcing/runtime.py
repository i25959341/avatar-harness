from __future__ import annotations

import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import InteractiveAvatarForcingConfig


@dataclass(frozen=True)
class InteractiveBlockResult:
    frames_rgb: np.ndarray
    total_seconds: float
    audio_seconds: float
    motion_seconds: float
    diffusion_seconds: float
    decode_seconds: float


@dataclass
class RuntimeSnapshot:
    kv_cache: Any
    latent_tail: torch.Tensor
    user_motion_tail: torch.Tensor
    avatar_audio: bytes
    user_audio: bytes
    frame_position: int
    cuda_rng_state: torch.Tensor


class InteractiveAvatarForcingRuntime:
    """Persistent blockwise runtime for TaekyungKi's AvatarForcing model."""

    def __init__(self, config: InteractiveAvatarForcingConfig) -> None:
        self.config = config
        self.agent: Any | None = None
        self.model: Any | None = None
        self._s_r: torch.Tensor | None = None
        self._r_s: torch.Tensor | None = None
        self._s_r_feats: list[torch.Tensor] | None = None
        self._source_rgb: np.ndarray | None = None
        self._latent_tail: torch.Tensor | None = None
        self._user_motion_tail: torch.Tensor | None = None
        self._avatar_audio = bytearray()
        self._user_audio = bytearray()
        self._frame_position = 0
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> float:
        if self._loaded:
            return 0.0
        self.config.validate()
        sys.path.insert(0, str(self.config.upstream_dir))

        import inference as upstream_inference
        from omegaconf import OmegaConf

        model_config = OmegaConf.load(self.config.upstream_dir / "configs" / "inference.yaml")
        model_config.rank = 0
        model_config.ngpus = 1
        model_config.mae_ckpt_path = str(
            self.config.upstream_dir / "pretrained_dir" / "motion_autoencoder.pth"
        )
        model_config.ckpt_path = str(
            self.config.upstream_dir / "pretrained_dir" / "flow_transformer.pth"
        )
        model_config.wav2vec_model_path = str(
            self.config.upstream_dir / "pretrained_dir" / "wav2vec2-base-960h"
        )

        started = time.perf_counter()
        upstream_inference.opt = model_config
        self.agent = upstream_inference.InferenceAgent(model_config)
        self.model = self.agent.G
        self._extend_rotary_positions()
        self._prepare_identity()
        self._loaded = True
        torch.cuda.synchronize()
        return time.perf_counter() - started

    def warmup(self) -> InteractiveBlockResult:
        self._require_loaded()
        with self._lock:
            result = self._initialize_stream()
            assert self._source_rgb is not None
            self._encode_user_motion([self._source_rgb] * self.config.block_frames)
            torch.cuda.synchronize()
            return result

    def snapshot(self) -> RuntimeSnapshot:
        self._require_stream()
        assert self.model is not None
        assert self._latent_tail is not None
        assert self._user_motion_tail is not None
        with self._lock:
            caches = []
            for self_cache, cross_cache in self.model.kv_cache:
                caches.append(
                    (
                        {name: value.clone() for name, value in self_cache.items()},
                        {name: value.clone() for name, value in cross_cache.items()},
                    )
                )
            return RuntimeSnapshot(
                kv_cache=caches,
                latent_tail=self._latent_tail.clone(),
                user_motion_tail=self._user_motion_tail.clone(),
                avatar_audio=bytes(self._avatar_audio),
                user_audio=bytes(self._user_audio),
                frame_position=self._frame_position,
                cuda_rng_state=torch.cuda.get_rng_state(),
            )

    def restore(self, snapshot: RuntimeSnapshot) -> None:
        self._require_stream()
        assert self.model is not None
        with self._lock:
            self.model.kv_cache = snapshot.kv_cache
            self._latent_tail = snapshot.latent_tail
            self._user_motion_tail = snapshot.user_motion_tail
            self._avatar_audio = bytearray(snapshot.avatar_audio)
            self._user_audio = bytearray(snapshot.user_audio)
            self._frame_position = snapshot.frame_position
            torch.cuda.set_rng_state(snapshot.cuda_rng_state)

    def generate_block(
        self,
        avatar_pcm_s16le: bytes,
        user_pcm_s16le: bytes,
        user_frames_rgb: Sequence[np.ndarray],
    ) -> InteractiveBlockResult:
        self._require_stream()
        if len(avatar_pcm_s16le) != self.config.bytes_per_block:
            raise ValueError("avatar audio must contain exactly one 400 ms block")
        if len(user_pcm_s16le) != self.config.bytes_per_block:
            raise ValueError("user audio must contain exactly one 400 ms block")
        with self._lock:
            return self._generate_block(avatar_pcm_s16le, user_pcm_s16le, user_frames_rgb)

    def _prepare_identity(self) -> None:
        assert self.agent is not None and self.model is not None
        processor = self.agent.data_processor
        face = processor.preprocess_face(str(self.config.source_image))
        self._source_rgb = np.ascontiguousarray(face)
        source = processor.transform(image=face)["image"].unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self._s_r, r_s_lambda, self._s_r_feats = self.model.encode_image_into_latent(source)
            self._r_s = self.model.motion_autoencoder.dec.direction(r_s_lambda)

    def _extend_rotary_positions(self) -> None:
        assert self.model is not None
        from models.avatarforcing.flow_transformer import precompute_freqs_cis

        blocks = self.model.flow_transformer.blocks
        attention = blocks[0].attn
        shared = precompute_freqs_cis(
            attention.head_dim,
            self.config.rotary_max_frames,
        ).to(attention.freqs_cis.device)
        for block in blocks:
            block.attn.freqs_cis = shared

    def _initialize_stream(self) -> InteractiveBlockResult:
        assert self.model is not None
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        self._avatar_audio = bytearray(self.config.history_bytes)
        self._user_audio = bytearray(self.config.history_bytes)
        self.model.denoising_step_list = torch.tensor(
            np.linspace(self.model.opt.num_train_timestep, 0, self.config.nfe - 1).tolist()
        )
        self.model.initialize_kv_cache(batch_size=1, dtype=torch.float32, device=0)

        total_started = time.perf_counter()
        audio_started = time.perf_counter()
        avatar_wa = self._encode_audio(bytes(self._avatar_audio))
        user_wa = self._encode_audio(bytes(self._user_audio))
        audio_seconds = time.perf_counter() - audio_started
        user_motion = torch.zeros_like(avatar_wa)

        x_t = torch.randn(1, self.config.history_frames, self.model.opt.dim_w, device="cuda")
        condition = self.model.prepare_cfg_condition(
            avatar_wa,
            user_wa,
            user_motion,
            self._r_s,
            seq_len=self.config.history_frames,
            context_len=0,
        )
        diffusion_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for index, timestep in enumerate(self.model.denoising_step_list):
                x_t = self.model.solve_cfg(
                    B=1,
                    index=index,
                    current_timestep=timestep,
                    x_t=x_t,
                    precomputed_c=condition[0],
                    precomputed_wr=condition[1],
                    precomputed_adaLN=condition[2],
                    start_pos=0,
                    context_len=0,
                    use_kv_cache=False,
                    a_cfg_scale=2.0,
                    u_cfg_scale=1.0,
                )
            self.model.update_kv_cache(x_t, *condition, start_pos=0)
        torch.cuda.synchronize()
        diffusion_seconds = time.perf_counter() - diffusion_started
        self._latent_tail = x_t[:, -self.config.context_frames :].detach()
        self._user_motion_tail = user_motion[:, -self.config.context_frames :].detach()
        self._frame_position = self.config.history_frames

        decode_started = time.perf_counter()
        frames = self._decode(x_t[:, -self.config.block_frames :])
        decode_seconds = time.perf_counter() - decode_started
        return InteractiveBlockResult(
            frames,
            time.perf_counter() - total_started,
            audio_seconds,
            0.0,
            diffusion_seconds,
            decode_seconds,
        )

    def _generate_block(
        self,
        avatar_pcm_s16le: bytes,
        user_pcm_s16le: bytes,
        user_frames_rgb: Sequence[np.ndarray],
    ) -> InteractiveBlockResult:
        assert self.model is not None
        assert self._latent_tail is not None
        assert self._user_motion_tail is not None
        total_started = time.perf_counter()
        self._append_audio(self._avatar_audio, avatar_pcm_s16le)
        self._append_audio(self._user_audio, user_pcm_s16le)

        audio_started = time.perf_counter()
        avatar_wa = self._encode_audio(bytes(self._avatar_audio))[:, -12:]
        user_wa = self._encode_audio(bytes(self._user_audio))[:, -12:]
        audio_seconds = time.perf_counter() - audio_started

        motion_started = time.perf_counter()
        new_user_motion = self._encode_user_motion(user_frames_rgb)
        user_motion = torch.cat([self._user_motion_tail, new_user_motion], dim=1)
        motion_seconds = time.perf_counter() - motion_started

        noise = torch.randn(1, self.config.block_frames, self.model.opt.dim_w, device="cuda")
        x_t = torch.cat([self._latent_tail, noise], dim=1)
        start_pos = self._frame_position - self.config.context_frames
        condition = self.model.prepare_cfg_condition(
            avatar_wa,
            user_wa,
            user_motion,
            self._r_s,
            seq_len=self.config.block_frames,
            context_len=self.config.context_frames,
        )

        diffusion_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for index, timestep in enumerate(self.model.denoising_step_list):
                x_t = self.model.solve_cfg(
                    B=1,
                    index=index,
                    current_timestep=timestep,
                    x_t=x_t,
                    precomputed_c=condition[0],
                    precomputed_wr=condition[1],
                    precomputed_adaLN=condition[2],
                    start_pos=start_pos,
                    context_len=self.config.context_frames,
                    use_kv_cache=True,
                    a_cfg_scale=2.0,
                    u_cfg_scale=1.0,
                )
            self.model.update_kv_cache(x_t, *condition, start_pos=start_pos)
        torch.cuda.synchronize()
        diffusion_seconds = time.perf_counter() - diffusion_started

        self._latent_tail = x_t[:, -self.config.context_frames :].detach()
        self._user_motion_tail = new_user_motion[:, -self.config.context_frames :].detach()
        self._frame_position += self.config.block_frames
        decode_started = time.perf_counter()
        frames = self._decode(x_t[:, -self.config.block_frames :])
        decode_seconds = time.perf_counter() - decode_started
        return InteractiveBlockResult(
            frames,
            time.perf_counter() - total_started,
            audio_seconds,
            motion_seconds,
            diffusion_seconds,
            decode_seconds,
        )

    def _encode_audio(self, pcm_s16le: bytes) -> torch.Tensor:
        assert self.agent is not None and self.model is not None
        samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        values = self.agent.data_processor.wav2vec_preprocessor(
            samples,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        ).input_values.to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model.audio_encoder.inference(values, seq_len=self.config.history_frames)

    def _encode_user_motion(self, frames_rgb: Sequence[np.ndarray]) -> torch.Tensor:
        assert self.model is not None
        if not frames_rgb:
            assert self._user_motion_tail is not None
            return torch.zeros(
                (1, self.config.block_frames, self._user_motion_tail.shape[-1]),
                device="cuda",
                dtype=self._user_motion_tail.dtype,
            )
        frames = list(frames_rgb[-self.config.block_frames :])
        while len(frames) < self.config.block_frames:
            frames.insert(0, frames[0])
        tensors = [
            torch.from_numpy(np.ascontiguousarray(frame))
            .permute(2, 0, 1)
            .float()
            .div(127.5)
            .sub(1.0)
            .unsqueeze(0)
            for frame in frames
        ]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model.encode_user_motion(tensors)

    def _decode(self, latent: torch.Tensor) -> np.ndarray:
        assert self.model is not None
        assert self._s_r is not None and self._r_s is not None and self._s_r_feats is not None
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoded = self.model.decode_latent_into_image(
                self._s_r,
                self._r_s,
                self._s_r_feats,
                latent,
            )["d_hat"]
        torch.cuda.synchronize()
        return decoded.clamp(-1, 1).add(1).mul(127.5).byte().cpu().permute(0, 2, 3, 1).numpy()

    def _append_audio(self, history: bytearray, block: bytes) -> None:
        history.extend(block)
        del history[: len(history) - self.config.history_bytes]

    def _require_loaded(self) -> None:
        if not self._loaded or self.model is None:
            raise RuntimeError("interactive AvatarForcing runtime is not loaded")

    def _require_stream(self) -> None:
        self._require_loaded()
        if self._latent_tail is None:
            raise RuntimeError("warmup() must be called before streaming")
