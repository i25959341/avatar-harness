#!/usr/bin/env python3
"""Compare generated and cached-idle handoff strategies for FlashHead output.

This compositor reuses an existing FlashHead interruption result. It tests visual
handoffs only; it does not generate transitions from live model state or connect
to LiveKit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
FPS = 25.0
OUTPUT_SIZE = (512, 512)
DEFAULT_GENERATED = (
    ROOT_DIR / "outputs" / "flashhead" / "interruption" / "flashhead_interruption_raw.mp4"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "flashhead" / "idle_strategy_comparison"


@dataclass(frozen=True)
class IdleRegion:
    name: str
    start: float
    end: float


IDLE_REGIONS = (
    IdleRegion("idle_before", 0.0, 1.5),
    IdleRegion("interrupted_idle", 4.0, 5.0),
    IdleRegion("idle_after", 8.0, 9.0),
)
BOUNDARIES = (
    ("idle_to_talk_1", 1.5, "idle_to_talk"),
    ("talk_1_to_idle", 4.0, "talk_to_idle"),
    ("idle_to_talk_2", 5.0, "idle_to_talk"),
    ("talk_2_to_idle", 8.0, "talk_to_idle"),
)


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def decode_video(path: Path, *, target_size: tuple[int, int]) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or source_fps <= 0:
        capture.release()
        raise RuntimeError(f"could not open video: {path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    return frames, source_fps


def resample_loop(frames: list[np.ndarray], source_fps: float, count: int) -> list[np.ndarray]:
    duration = len(frames) / source_fps
    return [
        frames[round(((index / FPS) % duration) * source_fps) % len(frames)]
        for index in range(count)
    ]


def comparison_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def match_idle_regions(
    generated: list[np.ndarray], idle_loop: list[np.ndarray]
) -> tuple[list[np.ndarray], dict[str, int]]:
    timeline = [frame.copy() for frame in generated]
    offsets: dict[str, int] = {}
    idle_gray = [comparison_gray(frame) for frame in idle_loop]

    for region in IDLE_REGIONS:
        start = round(region.start * FPS)
        end = min(round(region.end * FPS), len(generated))
        length = end - start
        target_start = comparison_gray(generated[start])
        target_end = comparison_gray(generated[end - 1])
        best_offset = min(
            range(len(idle_loop)),
            key=lambda offset: float(
                np.mean(np.square(idle_gray[offset] - target_start))
                + np.mean(np.square(idle_gray[(offset + length - 1) % len(idle_loop)] - target_end))
            ),
        )
        offsets[region.name] = best_offset
        for index in range(start, end):
            timeline[index] = idle_loop[(best_offset + index - start) % len(idle_loop)].copy()
    return timeline, offsets


def blend(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(first, 1.0 - alpha, second, alpha, 0)


def make_crossfade(
    generated: list[np.ndarray], hard_switch: list[np.ndarray], transition_frames: int
) -> list[np.ndarray]:
    output = [frame.copy() for frame in hard_switch]
    for _, timestamp, direction in BOUNDARIES:
        boundary = round(timestamp * FPS)
        if direction == "idle_to_talk":
            for step in range(transition_frames):
                index = boundary - transition_frames + step
                output[index] = blend(
                    hard_switch[index], generated[index], (step + 1) / transition_frames
                )
        else:
            for step in range(transition_frames):
                index = boundary + step
                output[index] = blend(
                    generated[index], hard_switch[index], (step + 1) / transition_frames
                )
    return output


def make_generated_bridge(
    generated: list[np.ndarray], hard_switch: list[np.ndarray], transition_frames: int
) -> list[np.ndarray]:
    output = [frame.copy() for frame in hard_switch]
    outer_blend = max(2, transition_frames // 2)
    for _, timestamp, direction in BOUNDARIES:
        boundary = round(timestamp * FPS)
        if direction == "idle_to_talk":
            start = boundary - transition_frames
            for index in range(start, boundary):
                output[index] = generated[index].copy()
            for step in range(outer_blend):
                index = start + step
                output[index] = blend(
                    hard_switch[index], generated[index], (step + 1) / outer_blend
                )
        else:
            end = min(boundary + transition_frames, len(output))
            for index in range(boundary, end):
                output[index] = generated[index].copy()
            blend_start = max(boundary, end - outer_blend)
            for step, index in enumerate(range(blend_start, end)):
                output[index] = blend(
                    generated[index], hard_switch[index], (step + 1) / outer_blend
                )
    return output


def write_video(frames: list[np.ndarray], audio_source: Path, output: Path) -> None:
    if not frames:
        raise RuntimeError("cannot write an empty frame sequence")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    frame_size = (frames[0].shape[1], frames[0].shape[0])
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        frame_size,
    )
    if not writer.isOpened():
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("could not initialize video writer")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(temporary_path),
                "-i",
                str(audio_source),
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
            ],
            check=True,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (labeled.shape[1], 52), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.7, labeled, 0.3, 0, labeled)
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.putText(
        labeled,
        label,
        ((labeled.shape[1] - size[0]) // 2, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_comparison_grid(strategies: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    labels = list(strategies)
    count = min(len(frames) for frames in strategies.values())
    output: list[np.ndarray] = []
    for index in range(count):
        tiles = [label_frame(strategies[label][index], label) for label in labels]
        output.append(np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:]))))
    return output


def frame_deltas(frames: list[np.ndarray]) -> np.ndarray:
    grays = [comparison_gray(frame) for frame in frames]
    return np.asarray(
        [
            float(np.mean(cv2.absdiff(grays[index - 1], grays[index])))
            for index in range(1, len(grays))
        ]
    )


def strategy_metrics(frames: list[np.ndarray], transition_frames: int) -> dict[str, Any]:
    deltas = frame_deltas(frames)
    transitions: dict[str, float] = {}
    transition_windows: dict[str, dict[str, float]] = {}
    for name, timestamp, _ in BOUNDARIES:
        index = min(max(round(timestamp * FPS) - 1, 0), len(deltas) - 1)
        transitions[name] = float(deltas[index])
        window_start = max(index - transition_frames, 0)
        window_end = min(index + transition_frames + 1, len(deltas))
        window = deltas[window_start:window_end]
        transition_windows[name] = {
            "mean": float(np.mean(window)),
            "max": float(np.max(window)),
        }

    idle_values: list[float] = []
    for region in IDLE_REGIONS:
        start = max(round(region.start * FPS), 1) - 1
        end = min(round(region.end * FPS) - 1, len(deltas))
        idle_values.extend(deltas[start:end])
    return {
        "mean_idle_motion": float(np.mean(idle_values)),
        "mean_transition_delta": float(np.mean(list(transitions.values()))),
        "max_transition_delta": max(transitions.values()),
        "transition_deltas": transitions,
        "transition_windows": transition_windows,
        "max_transition_window_delta": max(values["max"] for values in transition_windows.values()),
        "p99_all_frame_delta": float(np.percentile(deltas, 99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FlashHead idle handoff strategies.")
    parser.add_argument("--generated", type=root_path, default=DEFAULT_GENERATED)
    parser.add_argument("--idle-clip", type=root_path, default=ROOT_DIR / "assets" / "idle.mp4")
    parser.add_argument("--output-dir", type=root_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transition-frames", type=int, default=8)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.transition_frames < 2:
        raise RuntimeError("--transition-frames must be at least 2")
    generated, generated_fps = decode_video(args.generated, target_size=OUTPUT_SIZE)
    if not np.isclose(generated_fps, FPS):
        raise RuntimeError(f"generated video must be {FPS:g} FPS, got {generated_fps:g}")
    idle_source, idle_fps = decode_video(args.idle_clip, target_size=OUTPUT_SIZE)
    idle_loop = resample_loop(idle_source, idle_fps, len(idle_source) * 25 // 24)

    hard_switch, offsets = match_idle_regions(generated, idle_loop)
    strategies = {
        "1 GENERATED IDLE": [frame.copy() for frame in generated],
        "2 HARD IDLE CLIP": hard_switch,
        "3 8-FRAME CROSSFADE": make_crossfade(generated, hard_switch, args.transition_frames),
        "4 GENERATED BRIDGE": make_generated_bridge(generated, hard_switch, args.transition_frames),
    }

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "1 GENERATED IDLE": "01_generated_idle.mp4",
        "2 HARD IDLE CLIP": "02_hard_idle_clip.mp4",
        "3 8-FRAME CROSSFADE": "03_idle_clip_crossfade.mp4",
        "4 GENERATED BRIDGE": "04_generated_bridge.mp4",
    }
    outputs: dict[str, str] = {}
    for name, frames in strategies.items():
        path = output_dir / filenames[name]
        write_video(frames, args.generated, path)
        outputs[name] = str(path)

    comparison = output_dir / "idle_strategy_side_by_side.mp4"
    grid_frames = make_comparison_grid(strategies)
    write_video(grid_frames, args.generated, comparison)

    metrics = {
        name: strategy_metrics(frames, args.transition_frames)
        for name, frames in strategies.items()
    }
    report = {
        "scope": "offline visual composition; no live FlashHead state handoff or LiveKit",
        "generated_source": str(args.generated),
        "idle_clip": str(args.idle_clip),
        "idle_clip_source_fps": idle_fps,
        "idle_loop_frames_at_25fps": len(idle_loop),
        "transition_frames": args.transition_frames,
        "transition_duration_ms": args.transition_frames / FPS * 1000,
        "matched_idle_offsets": offsets,
        "outputs": outputs,
        "comparison": str(comparison),
        "metrics": metrics,
        "interpretation": {
            "mean_idle_motion": "higher means more visible idle movement",
            "transition_delta": "lower means a smaller single-frame handoff jump",
        },
    }
    report_path = output_dir / "idle_strategy_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
