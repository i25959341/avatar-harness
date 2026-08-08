#!/usr/bin/env python3
"""Render an IMTalker idle-to-speech-to-idle demonstration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "avatar_models" / "imtalker"))

from interactive_avatar.engine import IMTalkerEngine  # noqa: E402


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=root_path,
        default=ROOT_DIR / "avatar_models" / "imtalker" / "assets" / "source_1.png",
    )
    parser.add_argument(
        "--cache",
        type=root_path,
        default=ROOT_DIR / "outputs" / "cache" / "imtalker_idle.pt",
    )
    parser.add_argument(
        "--audio",
        type=root_path,
        default=ROOT_DIR / "avatar_models" / "imtalker" / "assets" / "audio_1.wav",
    )
    parser.add_argument(
        "--output",
        type=root_path,
        default=ROOT_DIR / "outputs" / "imtalker" / "demo.mp4",
    )
    parser.add_argument("--idle-seconds", type=float, default=1.0)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} not found: {path}")


def main() -> int:
    args = parse_args()
    require_file(args.source, "source image")
    require_file(args.cache, "idle cache")
    require_file(args.audio, "input audio")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples, _ = librosa.load(args.audio, sr=16_000, mono=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    duration = len(samples) / 16_000

    engine = IMTalkerEngine(
        source_image_path=str(args.source),
        idle_cache_path=str(args.cache),
        output_url=str(args.output),
    )
    engine.start()
    try:
        time.sleep(args.idle_seconds)
        for offset in range(0, len(pcm), 6400):
            engine.push_audio(pcm[offset : offset + 6400])
        engine.end_audio()
        time.sleep(duration + args.idle_seconds + 2.0)
    finally:
        engine.stop()

    print(f"Rendered {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
