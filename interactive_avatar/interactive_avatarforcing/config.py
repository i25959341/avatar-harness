from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class InteractiveAvatarForcingConfig:
    source_image: Path = ROOT_DIR / "third_party" / "InteractiveAvatarForcing" / "data" / "rumi.jpg"
    width: int = 512
    height: int = 512
    fps: int = 25
    sample_rate: int = 16_000
    block_frames: int = 10
    context_frames: int = 2
    history_frames: int = 50
    rotary_max_frames: int = 262_144
    nfe: int = 4
    seed: int = 25
    output_queue_frames: int = 20
    camera_stale_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> InteractiveAvatarForcingConfig:
        defaults = cls()
        source = Path(
            os.environ.get(
                "INTERACTIVE_AVATARFORCING_SOURCE_IMAGE",
                str(defaults.source_image),
            )
        ).expanduser()
        if not source.is_absolute():
            source = (ROOT_DIR / source).resolve()
        return cls(
            source_image=source,
            nfe=int(os.environ.get("INTERACTIVE_AVATARFORCING_NFE", defaults.nfe)),
            seed=int(os.environ.get("INTERACTIVE_AVATARFORCING_SEED", defaults.seed)),
        )

    @property
    def upstream_dir(self) -> Path:
        return ROOT_DIR / "third_party" / "InteractiveAvatarForcing"

    @property
    def python(self) -> Path:
        return self.upstream_dir / ".venv" / "bin" / "python"

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate // self.fps

    @property
    def samples_per_block(self) -> int:
        return self.samples_per_frame * self.block_frames

    @property
    def bytes_per_block(self) -> int:
        return self.samples_per_block * 2

    @property
    def history_bytes(self) -> int:
        return self.history_frames * self.samples_per_frame * 2

    @property
    def block_seconds(self) -> float:
        return self.block_frames / self.fps

    def validate(self) -> None:
        if self.sample_rate % self.fps:
            raise RuntimeError("sample rate must divide evenly by video FPS")
        if self.block_frames != 10 or self.context_frames != 2 or self.history_frames != 50:
            raise RuntimeError("the released model requires 10-frame blocks and 2/50-frame context")
        if self.nfe < 2:
            raise RuntimeError("NFE must be at least 2")
        if self.rotary_max_frames <= self.history_frames:
            raise RuntimeError("rotary position capacity must exceed the initial history")
        if self.camera_stale_seconds <= 0:
            raise RuntimeError("camera stale timeout must be positive")
        required = (
            self.source_image,
            self.upstream_dir / "inference.py",
            self.upstream_dir / "pretrained_dir" / "motion_autoencoder.pth",
            self.upstream_dir / "pretrained_dir" / "flow_transformer.pth",
            self.upstream_dir / "pretrained_dir" / "wav2vec2-base-960h" / "config.json",
            self.python,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError("missing interactive AvatarForcing assets: " + ", ".join(missing))
