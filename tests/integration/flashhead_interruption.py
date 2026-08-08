#!/usr/bin/env python3
"""Exercise FlashHead across idle, speech, interruption, and resumed speech.

This is an offline model-state diagnostic. It does not exercise LiveKit queues,
generation cancellation, WebRTC delivery, or receiver-side synchronization.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT_DIR / "outputs" / "flashhead" / "interruption"
FPS = 25.0
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Segment:
    name: str
    label: str
    start: float
    end: float
    expects_speech: bool


SEGMENTS = (
    Segment("idle_before", "IDLE", 0.0, 1.5, False),
    Segment("talk_1", "TALK 1", 1.5, 4.0, True),
    Segment("interrupted", "INTERRUPTED / IDLE", 4.0, 5.0, False),
    Segment("talk_2", "TALK 2", 5.0, 8.0, True),
    Segment("idle_after", "IDLE", 8.0, 9.0, False),
)


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def run_command(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=True,
        capture_output=capture,
    )


def make_fixture(source_audio: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_filter = (
        "anullsrc=r=16000:cl=mono:d=1.5[idle0];"
        "[0:a]atrim=start=0:end=2.5,asetpts=PTS-STARTPTS[talk0];"
        "anullsrc=r=16000:cl=mono:d=1.0[idle1];"
        "[0:a]atrim=start=4:end=7,asetpts=PTS-STARTPTS[talk1];"
        "anullsrc=r=16000:cl=mono:d=1.0[idle2];"
        "[idle0][talk0][idle1][talk1][idle2]concat=n=5:v=0:a=1[out]"
    )
    run_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source_audio),
            "-filter_complex",
            audio_filter,
            "-map",
            "[out]",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def generate(source_image: Path, fixture_audio: Path, output: Path) -> None:
    run_command(
        [
            sys.executable,
            str(ROOT_DIR / "examples" / "generate_flashhead_video.py"),
            "--source",
            str(source_image),
            "--audio",
            str(fixture_audio),
            "--output",
            str(output),
        ]
    )


def probe_duration(path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture=True,
    )
    return float(result.stdout.decode().strip())


def decode_audio(path: Path) -> np.ndarray:
    result = run_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-",
        ],
        capture=True,
    )
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def audio_rms_by_segment(audio: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    margin = int(0.15 * SAMPLE_RATE)
    for segment in SEGMENTS:
        start = int(segment.start * SAMPLE_RATE) + margin
        end = int(segment.end * SAMPLE_RATE) - margin
        samples = audio[start:end]
        values[segment.name] = float(np.sqrt(np.mean(np.square(samples))))
    return values


def decode_frames(path: Path) -> tuple[list[np.ndarray], list[float]]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    deltas: list[float] = []
    previous_gray: np.ndarray | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                deltas.append(float(cv2.absdiff(gray, previous_gray).mean()))
            previous_gray = gray
    finally:
        capture.release()
    if len(frames) < 2:
        raise RuntimeError("generated video has fewer than two decodable frames")
    return frames, deltas


def motion_report(deltas: list[float]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    all_deltas = np.asarray(deltas)
    boundaries: dict[str, float] = {}
    for segment in SEGMENTS[1:]:
        # deltas[i] compares video frame i with frame i + 1.
        index = min(max(round(segment.start * FPS) - 1, 0), len(deltas) - 1)
        boundaries[segment.name] = deltas[index]

    segment_motion: dict[str, dict[str, float]] = {}
    for segment in SEGMENTS:
        start = max(round(segment.start * FPS), 1) - 1
        end = min(round(segment.end * FPS) - 1, len(deltas))
        values = np.asarray(deltas[start:end])
        segment_motion[segment.name] = {
            "mean_frame_delta": float(np.mean(values)),
            "max_frame_delta": float(np.max(values)),
        }

    summary = {
        "mean_frame_delta": float(np.mean(all_deltas)),
        "p99_frame_delta": float(np.percentile(all_deltas, 99)),
        "max_frame_delta": float(np.max(all_deltas)),
        "max_transition_delta": max(boundaries.values()),
    }
    return summary | {"transition_deltas": boundaries}, segment_motion


def annotate_video(source: Path, output: Path) -> None:
    temporary_video = output.with_suffix(".video.avi")
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        (width, height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("could not initialize annotated video writer")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / FPS
            segment = next(
                (item for item in SEGMENTS if item.start <= timestamp < item.end),
                SEGMENTS[-1],
            )
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 58), (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            text_size = cv2.getTextSize(segment.label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(
                frame,
                segment.label,
                ((width - text_size[0]) // 2, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    try:
        run_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(temporary_video),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-c:a",
                "copy",
                str(output),
            ]
        )
    finally:
        temporary_video.unlink(missing_ok=True)


def make_contact_sheet(frames: list[np.ndarray], output: Path) -> None:
    tiles: list[np.ndarray] = []
    for segment in SEGMENTS:
        midpoint = (segment.start + segment.end) / 2
        frame = frames[min(round(midpoint * FPS), len(frames) - 1)].copy()
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (256, 36), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            segment.label,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(frame)
    cv2.imwrite(str(output), np.hstack(tiles))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline FlashHead interruption test.")
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
    parser.add_argument("--output-dir", type=root_path, default=DEFAULT_DIR)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="analyze existing files in --output-dir",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir: Path = args.output_dir
    fixture = output_dir / "interruption_input.wav"
    raw_video = output_dir / "flashhead_interruption_raw.mp4"
    review_video = output_dir / "flashhead_interruption_review.mp4"
    contact_sheet = output_dir / "flashhead_interruption_contact.jpg"
    report_path = output_dir / "flashhead_interruption_report.json"

    if not args.skip_generation:
        make_fixture(args.audio, fixture)
        generate(args.source, fixture, raw_video)
    elif not fixture.is_file() or not raw_video.is_file():
        raise RuntimeError(f"existing fixture or video missing from {output_dir}")

    duration = probe_duration(raw_video)
    audio = decode_audio(raw_video)
    frames, deltas = decode_frames(raw_video)
    rms = audio_rms_by_segment(audio)
    motion, segment_motion = motion_report(deltas)
    annotate_video(raw_video, review_video)
    make_contact_sheet(frames, contact_sheet)

    continuity_limit = max(2.0, motion["p99_frame_delta"] * 2.0)
    checks = {
        "duration_within_one_frame": abs(duration - SEGMENTS[-1].end) <= 1 / FPS,
        "frame_count_within_one_frame": abs(len(frames) - SEGMENTS[-1].end * FPS) <= 1,
        "speech_present_in_both_turns": rms["talk_1"] > 0.01 and rms["talk_2"] > 0.01,
        "silence_during_idle_and_interruption": all(
            rms[name] < 0.001 for name in ("idle_before", "interrupted", "idle_after")
        ),
        "no_large_visual_transition": motion["max_transition_delta"] <= continuity_limit,
    }
    report: dict[str, Any] = {
        "scope": "offline FlashHead model-state test; not a LiveKit cancellation test",
        "timeline": [asdict(segment) for segment in SEGMENTS],
        "interruption_at_seconds": 4.0,
        "resumed_speech_at_seconds": 5.0,
        "input_audio": str(fixture),
        "raw_video": str(raw_video),
        "review_video": str(review_video),
        "contact_sheet": str(contact_sheet),
        "duration_seconds": duration,
        "decoded_frames": len(frames),
        "audio_rms": rms,
        "motion": motion,
        "segment_motion": segment_motion,
        "continuity_limit": continuity_limit,
        "checks": checks,
        "passed": all(checks.values()),
        "not_measured": [
            "LiveKit queue flush",
            "generation cancellation latency",
            "stale frame delivery at a remote participant",
            "receiver-side A/V skew",
        ],
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
        print("FlashHead interruption diagnostic failed", file=sys.stderr)
        return 1
    print("FlashHead offline interruption diagnostic passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
