#!/usr/bin/env python3
"""Install Interactive AvatarForcing and prepare its bundled evaluation sample."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT_DIR / "third_party" / "InteractiveAvatarForcing"
VENV_DIR = UPSTREAM_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"


def run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-sample-preprocess", action="store_true")
    args = parser.parse_args()

    if not UPSTREAM_DIR.joinpath("inference.py").is_file():
        raise RuntimeError(
            "Interactive AvatarForcing submodule is missing; run git submodule update --init"
        )

    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        [str(VENV_DIR / "bin"), environment.get("PATH", "")]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    if not args.skip_install:
        if not VENV_PYTHON.is_file():
            run([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT_DIR, environment=environment)
        run(
            [str(VENV_PYTHON), "-m", "ensurepip", "--upgrade"],
            cwd=UPSTREAM_DIR,
            environment=environment,
        )
        run(
            [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
            cwd=UPSTREAM_DIR,
            environment=environment,
        )
        run(
            [
                str(VENV_PYTHON),
                "-m",
                "pip",
                "install",
                "torch==2.7.0",
                "torchvision==0.22.0",
                "torchaudio==2.7.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            cwd=UPSTREAM_DIR,
            environment=environment,
        )
        run(
            [str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=UPSTREAM_DIR,
            environment=environment,
        )
        run(
            [
                str(VENV_PYTHON),
                "-m",
                "pip",
                "install",
                "livekit-agents>=1.6,<1.7",
            ],
            cwd=UPSTREAM_DIR,
            environment=environment,
        )

    if not VENV_PYTHON.is_file():
        raise RuntimeError("Interactive AvatarForcing environment is missing; omit --skip-install")

    if not args.skip_download:
        run(["bash", "download_weights.sh"], cwd=UPSTREAM_DIR, environment=environment)

    if not args.skip_sample_preprocess:
        sample_output = ROOT_DIR / "outputs" / "interactive-avatarforcing" / "input"
        sample_output.mkdir(parents=True, exist_ok=True)
        run(
            [
                str(VENV_PYTHON),
                "preprocess_user_video.py",
                "--user_video_path",
                str(UPSTREAM_DIR / "data" / "user.mp4"),
                "--output_path",
                str(sample_output),
                "--pad_ratio",
                "1.0",
            ],
            cwd=UPSTREAM_DIR,
            environment=environment,
        )

    validation_environment = environment.copy()
    validation_environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT_DIR / "interactive_avatar" / "interactive_avatarforcing_compat"),
            str(UPSTREAM_DIR),
            str(ROOT_DIR),
        ]
    )
    run(
        [
            str(VENV_PYTHON),
            "-c",
            (
                "from interactive_avatar.interactive_avatarforcing.config import "
                "InteractiveAvatarForcingConfig; InteractiveAvatarForcingConfig().validate()"
            ),
        ],
        cwd=ROOT_DIR,
        environment=validation_environment,
    )
    print("Interactive AvatarForcing setup is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
