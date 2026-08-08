#!/usr/bin/env python3
"""Install AVTR-1 dependencies, download artifacts, and build TensorRT engines."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from interactive_avatar.avtr1 import Avtr1Config  # noqa: E402
from interactive_avatar.environment import load_env_file  # noqa: E402


def run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_env_file(ROOT_DIR / ".env.local")
    config = Avtr1Config.from_env()
    if not config.avtr1_dir.joinpath("pixi.toml").is_file():
        raise RuntimeError("AVTR-1 submodule is missing; run git submodule update --init")
    if not config.pixi.is_file():
        raise RuntimeError(
            "Pixi is missing. Install it under .pixi-local or follow docs/backends/avtr1.md"
        )

    environment = os.environ.copy()
    environment.setdefault("AVTR1_LOCAL_STORAGE", str(config.avtr1_dir / "artifacts"))
    pixi = str(config.pixi)
    if not args.skip_install:
        run([pixi, "install", "-e", "renderer"], cwd=config.avtr1_dir, environment=environment)
    if not args.skip_download:
        if not environment.get("HF_TOKEN") and not Path.home().joinpath(
            ".cache/huggingface/token"
        ).is_file():
            raise RuntimeError(
                "Hugging Face authentication is required. Accept access to "
                "avaturn-live/avtr-1, then run `pixi run -e renderer hf auth login` "
                "inside third_party/avtr-1 or set HF_TOKEN in .env.local."
            )
        run(
            [
                pixi,
                "run",
                "-e",
                "renderer",
                "python",
                "scripts/download_artifacts.py",
                "--workers",
                str(args.workers),
            ],
            cwd=config.avtr1_dir,
            environment=environment,
        )
    if not args.skip_build:
        run(
            [pixi, "run", "-e", "renderer", "python", "scripts/build_engines.py"],
            cwd=config.avtr1_dir,
            environment=environment,
        )
    config.validate_setup()
    print("AVTR-1 setup is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
