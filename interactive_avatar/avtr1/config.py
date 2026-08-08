from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Avtr1Config:
    avtr1_dir: Path = ROOT_DIR / "third_party" / "avtr-1"
    renderer_url: str = "http://127.0.0.1:8000"
    avatar_id: str = "maria"
    background_id: str = "plain_white"
    width: int = 1280
    height: int = 720
    fps: int = 25
    sample_rate: int = 16_000
    chunk_frames: int = 5
    current_samples: int = 3_200
    future_samples: int = 3_280
    output_queue_frames: int = 10
    renderer_start_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> Avtr1Config:
        defaults = cls()
        return cls(
            renderer_url=os.environ.get("AVTR1_RENDERER_URL", defaults.renderer_url),
            avatar_id=os.environ.get("AVTR1_AVATAR_ID", defaults.avatar_id),
            background_id=os.environ.get(
                "AVTR1_BACKGROUND_ID", defaults.background_id
            ),
        )

    @property
    def pixi(self) -> Path:
        return ROOT_DIR / ".pixi-local" / "bin" / "pixi"

    @property
    def storage_root(self) -> Path:
        return self.avtr1_dir / "artifacts" / "main"

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate // self.fps

    @property
    def bytes_per_frame(self) -> int:
        return self.samples_per_frame * 2

    @property
    def chunk_seconds(self) -> float:
        return self.chunk_frames / self.fps

    def validate_setup(self) -> None:
        if self.sample_rate % self.fps:
            raise RuntimeError("sample rate must divide evenly by video FPS")
        required = (
            self.avtr1_dir / "pixi.toml",
            self.pixi,
            self.storage_root
            / "speech2motion_runtime_artifacts_cc"
            / "avtr1_encode_fp16.engine",
            self.storage_root
            / "speech2motion_runtime_artifacts_cc"
            / "avtr1_decode_fp16.engine",
            self.storage_root
            / "speech2motion_runtime_artifacts_cc"
            / "hubert_lbs_fp16.engine",
            self.storage_root / "renderer_runtime_artifacts_cc" / "decoder_b5_fp16.engine",
            self.storage_root
            / "renderer_runtime_artifacts_cc"
            / "warp_network_b5_fp16.engine",
            self.storage_root / "renderer_runtime_artifacts_cc" / "modnet_b5_fp16.engine",
            self.storage_root
            / "renderer_runtime_artifacts_cc"
            / "stitch_network_b5_fp16.engine",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError("missing AVTR-1 setup assets: " + ", ".join(missing))
