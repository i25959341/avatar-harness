#!/usr/bin/env python3
"""Render an AVTR-1 offline speech, listening, or idle sample."""

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


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def main() -> int:
    defaults = Avtr1Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=root_path)
    parser.add_argument("--listen", type=root_path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--avatar", default=defaults.avatar_id)
    parser.add_argument("--background", default=defaults.background_id)
    parser.add_argument(
        "--output",
        type=root_path,
        default=ROOT_DIR / "outputs" / "avtr1" / "offline.mp4",
    )
    args = parser.parse_args()
    if args.speech is None and args.listen is None and args.duration is None:
        parser.error("provide --speech, --listen, or --duration")

    load_env_file(ROOT_DIR / ".env.local")
    config = Avtr1Config.from_env()
    config.validate_setup()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.pixi),
        "run",
        "-e",
        "renderer",
        "python",
        "scripts/generate_offline.py",
        "--avatar",
        args.avatar,
        "--bg",
        args.background,
        "--out",
        str(args.output),
    ]
    if args.speech:
        command.extend(["--speech", str(args.speech)])
    if args.listen:
        command.extend(["--listen", str(args.listen)])
    if args.duration is not None:
        command.extend(["--duration", str(args.duration)])

    environment = os.environ.copy()
    environment.setdefault("AVTR1_LOCAL_STORAGE", str(config.avtr1_dir / "artifacts"))
    subprocess.run(command, cwd=config.avtr1_dir, env=environment, check=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
