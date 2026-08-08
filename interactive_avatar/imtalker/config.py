from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IMTalkerConfig:
    source_image: Path = ROOT_DIR / "avatar_models" / "imtalker" / "assets" / "source_1.png"
    idle_cache: Path = ROOT_DIR / "outputs" / "cache" / "imtalker_idle.pt"
    width: int = 512
    height: int = 512
    fps: int = 25
    sample_rate: int = 16_000
    min_chunk_frames: int = 10
    max_chunk_frames: int = 50
    default_chunk_frames: int = 25

    @classmethod
    def from_env(cls) -> IMTalkerConfig:
        def env_path(name: str, default: Path) -> Path:
            value = os.environ.get(name)
            if not value:
                return default
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()

        defaults = cls()
        return cls(
            source_image=env_path("IMTALKER_SOURCE_IMAGE", defaults.source_image),
            idle_cache=env_path("IMTALKER_IDLE_CACHE", defaults.idle_cache),
        )

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate // self.fps

    @property
    def bytes_per_frame(self) -> int:
        return self.samples_per_frame * 2

    def validate(self) -> None:
        required = (
            self.source_image,
            self.idle_cache,
            ROOT_DIR / "checkpoints" / "renderer.ckpt",
            ROOT_DIR / "checkpoints" / "generator.ckpt",
            ROOT_DIR / "checkpoints" / "wav2vec2-base-960h" / "config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("missing IMTalker assets: " + ", ".join(missing))
        if self.sample_rate % self.fps:
            raise RuntimeError("sample rate must divide evenly by video FPS")
