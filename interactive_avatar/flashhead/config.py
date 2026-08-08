from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FlashHeadConfig:
    flashhead_dir: Path = ROOT_DIR / "third_party" / "SoulX-FlashHead"
    source_image: Path = ROOT_DIR / "avatar_models" / "imtalker" / "assets" / "source_1.png"
    idle_video: Path = ROOT_DIR / "assets" / "idle.mp4"
    width: int = 512
    height: int = 512
    fps: int = 25
    sample_rate: int = 16_000
    chunk_frames: int = 24
    motion_frames: int = 9
    bridge_frames: int = 8
    interruption_transition: str = "generated"
    interruption_bridge_frames: int = 4
    audio_context_seconds: int = 8
    output_queue_frames: int = 48
    seed: int = 42

    @classmethod
    def from_env(cls) -> FlashHeadConfig:
        def env_path(name: str, default: Path) -> Path:
            value = os.environ.get(name)
            if not value:
                return default
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()

        defaults = cls()
        return cls(
            flashhead_dir=env_path("FLASHHEAD_DIR", defaults.flashhead_dir),
            source_image=env_path("FLASHHEAD_SOURCE_IMAGE", defaults.source_image),
            idle_video=env_path("FLASHHEAD_IDLE_VIDEO", defaults.idle_video),
            interruption_transition=os.environ.get(
                "FLASHHEAD_INTERRUPTION_TRANSITION", defaults.interruption_transition
            )
            .strip()
            .lower(),
            interruption_bridge_frames=int(
                os.environ.get(
                    "FLASHHEAD_INTERRUPTION_BRIDGE_FRAMES",
                    defaults.interruption_bridge_frames,
                )
            ),
        )

    @property
    def checkpoint_dir(self) -> Path:
        return self.flashhead_dir / "models" / "SoulX-FlashHead-1_3B"

    @property
    def wav2vec_dir(self) -> Path:
        return self.flashhead_dir / "models" / "wav2vec2-base-960h"

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate // self.fps

    @property
    def samples_per_chunk(self) -> int:
        return self.samples_per_frame * self.chunk_frames

    @property
    def bytes_per_chunk(self) -> int:
        return self.samples_per_chunk * 2

    def validate(self) -> None:
        required = (
            self.source_image,
            self.idle_video,
            self.checkpoint_dir / "Model_Lite" / "diffusion_pytorch_model.safetensors",
            self.checkpoint_dir / "VAE_LTX" / "diffusion_pytorch_model.safetensors",
            self.wav2vec_dir / "config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("missing FlashHead assets: " + ", ".join(missing))
        if self.sample_rate % self.fps:
            raise RuntimeError("sample rate must divide evenly by video FPS")
        if self.interruption_transition not in {"generated", "vae", "pixel", "hard"}:
            raise RuntimeError(
                "FLASHHEAD_INTERRUPTION_TRANSITION must be generated, vae, pixel, or hard"
            )
        if self.interruption_bridge_frames < 2:
            raise RuntimeError("FLASHHEAD_INTERRUPTION_BRIDGE_FRAMES must be at least 2")
