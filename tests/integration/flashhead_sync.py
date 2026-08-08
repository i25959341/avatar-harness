#!/usr/bin/env python3
"""Generate and validate an offline FlashHead audio/video timeline.

This checks local media accounting and decode integrity. It does not connect to a
LiveKit room and therefore cannot measure receiver-side WebRTC synchronization.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "flashhead" / "flashhead_sync_test.mp4"


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,size:stream=index,codec_type,codec_name,width,height,"
            "r_frame_rate,nb_frames,sample_rate,channels"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def stream(metadata: dict[str, Any], media_type: str) -> dict[str, Any]:
    try:
        return next(item for item in metadata["streams"] if item["codec_type"] == media_type)
    except StopIteration as error:
        raise RuntimeError(f"missing {media_type} stream") from error


def parse_rate(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def measure_motion(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    previous: np.ndarray | None = None
    deltas: list[float] = []
    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                deltas.append(float(cv2.absdiff(gray, previous).mean()))
            previous = gray
    finally:
        capture.release()

    if not deltas:
        raise RuntimeError("video contains fewer than two decodable frames")

    moving = sum(delta > 0.25 for delta in deltas)
    return {
        "decoded_frames": frame_count,
        "mean_frame_delta": float(np.mean(deltas)),
        "max_frame_delta": float(np.max(deltas)),
        "moving_frame_ratio": moving / len(deltas),
    }


def make_contact_sheet(video: Path, output: Path, duration: float) -> None:
    interval = max(duration / 4.0, 0.04)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps=1/{interval},scale=256:256,tile=4x1",
        "-frames:v",
        "1",
        str(output),
        "-y",
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline FlashHead sync diagnostic.")
    parser.add_argument(
        "--source",
        type=root_path,
        default=ROOT_DIR / "avatar_models" / "imtalker" / "assets" / "source_1.png",
    )
    parser.add_argument(
        "--audio",
        type=root_path,
        default=ROOT_DIR / "avatar_models" / "imtalker" / "assets" / "audio_1.wav",
    )
    parser.add_argument("--output", type=root_path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="validate an existing --output file without running FlashHead",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_generation:
        command = [
            sys.executable,
            str(ROOT_DIR / "examples" / "generate_flashhead_video.py"),
            "--source",
            str(args.source),
            "--audio",
            str(args.audio),
            "--output",
            str(output),
        ]
        subprocess.run(command, cwd=ROOT_DIR, check=True)
    elif not output.is_file():
        raise RuntimeError(f"output does not exist: {output}")

    input_metadata = probe(args.audio)
    output_metadata = probe(output)
    video = stream(output_metadata, "video")
    audio = stream(output_metadata, "audio")

    input_duration = float(input_metadata["format"]["duration"])
    output_duration = float(output_metadata["format"]["duration"])
    fps = parse_rate(video["r_frame_rate"])
    encoded_frames = int(video["nb_frames"])
    expected_frames = input_duration * fps
    duration_error_ms = abs(output_duration - input_duration) * 1000
    frame_error = abs(encoded_frames - expected_frames)

    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
    )
    motion = measure_motion(output)

    checks = {
        "duration_within_one_video_frame": duration_error_ms <= (1000 / fps),
        "frame_count_within_one_frame": frame_error <= 1.0,
        "video_is_512x512": video.get("width") == 512 and video.get("height") == 512,
        "video_is_25_fps": math.isclose(fps, 25.0),
        "audio_is_16khz_mono": audio.get("sample_rate") == "16000" and audio.get("channels") == 1,
        "all_frames_decode": motion["decoded_frames"] == encoded_frames,
        "video_is_animated": motion["moving_frame_ratio"] > 0.50
        and motion["mean_frame_delta"] > 0.10,
    }

    contact_sheet = output.with_name(f"{output.stem}_contact.jpg")
    report_path = output.with_name(f"{output.stem}_report.json")
    make_contact_sheet(output, contact_sheet, output_duration)

    report = {
        "scope": "offline FlashHead timeline; no LiveKit room or receiver",
        "source": str(args.source),
        "input_audio": str(args.audio),
        "output": str(output),
        "contact_sheet": str(contact_sheet),
        "input_duration_seconds": input_duration,
        "output_duration_seconds": output_duration,
        "duration_error_ms": duration_error_ms,
        "fps": fps,
        "encoded_frames": encoded_frames,
        "expected_frames": expected_frames,
        "frame_error": frame_error,
        "audio_sample_rate": int(audio["sample_rate"]),
        "audio_channels": int(audio["channels"]),
        "motion": motion,
        "checks": checks,
        "passed": all(checks.values()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        print("FlashHead sync diagnostic failed", file=sys.stderr)
        return 1
    print("FlashHead offline timeline diagnostic passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
