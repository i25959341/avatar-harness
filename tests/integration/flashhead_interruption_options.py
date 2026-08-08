#!/usr/bin/env python3
"""Render controlled FlashHead interruption handoff options for visual review."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
FLASHHEAD_DIR = ROOT_DIR / "third_party" / "SoulX-FlashHead"
FLASHHEAD_PYTHON = FLASHHEAD_DIR / ".venv" / "bin" / "python"
DEFAULT_GENERATED = (
    ROOT_DIR / "outputs" / "flashhead" / "interruption" / "flashhead_interruption_raw.mp4"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "flashhead" / "interruption_strategy_comparison"
FPS = 25.0
INTERRUPTION_SECONDS = 1.72
SOURCE_RESUME_SECONDS = 5.0
RESUME_SECONDS = 4.72
MOTION_FRAMES = 9
GENERATED_BRIDGE_FRAMES = 8
DIRECT_BRIDGE_FRAMES = 4
HYBRID_GENERATED_FRAMES = 4


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (ROOT_DIR / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=root_path, default=DEFAULT_GENERATED)
    parser.add_argument("--idle-clip", type=root_path, default=ROOT_DIR / "assets" / "idle.mp4")
    parser.add_argument("--output-dir", type=root_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--interruption-seconds", type=float, default=INTERRUPTION_SECONDS)
    parser.add_argument("--source-resume-seconds", type=float, default=SOURCE_RESUME_SECONDS)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    return parser.parse_args()


def in_flashhead_environment() -> bool:
    return Path(sys.prefix).resolve() == (FLASHHEAD_DIR / ".venv").resolve()


def reexec_in_flashhead_environment() -> int:
    if not FLASHHEAD_PYTHON.is_file():
        raise RuntimeError(f"FlashHead Python not found: {FLASHHEAD_PYTHON}")
    return subprocess.run(
        [str(FLASHHEAD_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], check=False
    ).returncode


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def nearest_idle_index(frame: np.ndarray, idle_frames: list[np.ndarray]) -> int:
    target = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (64, 64),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    return min(
        range(len(idle_frames)),
        key=lambda index: float(
            np.mean(
                np.square(
                    cv2.resize(
                        cv2.cvtColor(idle_frames[index], cv2.COLOR_BGR2GRAY),
                        (64, 64),
                        interpolation=cv2.INTER_AREA,
                    ).astype(np.float32)
                    - target
                )
            )
        ),
    )


def idle_sequence(idle_frames: list[np.ndarray], start: int, count: int) -> list[np.ndarray]:
    return [idle_frames[(start + offset) % len(idle_frames)].copy() for offset in range(count)]


def replace_interruption(
    generated: list[np.ndarray], replacement: list[np.ndarray]
) -> list[np.ndarray]:
    output = [frame.copy() for frame in generated]
    start = round(INTERRUPTION_SECONDS * FPS)
    end = round(RESUME_SECONDS * FPS)
    if len(replacement) != end - start:
        raise RuntimeError(f"expected {end - start} interruption frames, got {len(replacement)}")
    output[start:end] = replacement
    return output


def expanded_timeline(generated: list[np.ndarray]) -> list[np.ndarray]:
    interruption = round(INTERRUPTION_SECONDS * FPS)
    source_resume = round(SOURCE_RESUME_SECONDS * FPS)
    idle_count = round((RESUME_SECONDS - INTERRUPTION_SECONDS) * FPS)
    return (
        [frame.copy() for frame in generated[:interruption]]
        + [generated[interruption - 1].copy() for _ in range(idle_count)]
        + [frame.copy() for frame in generated[source_resume:]]
    )


def make_expanded_audio(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_filter = (
        f"[0:a]atrim=start=0:end={INTERRUPTION_SECONDS},asetpts=PTS-STARTPTS[first];"
        f"anullsrc=r=16000:cl=mono:d={RESUME_SECONDS - INTERRUPTION_SECONDS}[idle];"
        f"[0:a]atrim=start={SOURCE_RESUME_SECONDS},asetpts=PTS-STARTPTS[second];"
        "[first][idle][second]concat=n=3:v=0:a=1[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            audio_filter,
            "-map",
            "[out]",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
    )


def hard_cut(
    generated: list[np.ndarray], idle_frames: list[np.ndarray]
) -> tuple[list[np.ndarray], int]:
    boundary = round(INTERRUPTION_SECONDS * FPS)
    count = round((RESUME_SECONDS - INTERRUPTION_SECONDS) * FPS)
    idle_start = nearest_idle_index(generated[boundary - 1], idle_frames)
    return replace_interruption(generated, idle_sequence(idle_frames, idle_start, count)), idle_start


def direct_pixel_blend(
    generated: list[np.ndarray], idle_frames: list[np.ndarray]
) -> tuple[list[np.ndarray], int]:
    boundary = round(INTERRUPTION_SECONDS * FPS)
    count = round((RESUME_SECONDS - INTERRUPTION_SECONDS) * FPS)
    source = generated[boundary - 1]
    idle_start = nearest_idle_index(source, idle_frames)
    replacement = idle_sequence(idle_frames, idle_start, count)
    for step in range(DIRECT_BRIDGE_FRAMES):
        alpha = smoothstep((step + 1) / DIRECT_BRIDGE_FRAMES)
        replacement[step] = cv2.addWeighted(source, 1.0 - alpha, replacement[step], alpha, 0)
    return replace_interruption(generated, replacement), idle_start


def current_generated_bridge(
    generated: list[np.ndarray],
    idle_frames: list[np.ndarray],
    bridge: list[np.ndarray],
    generation_delay_frames: int,
) -> list[np.ndarray]:
    boundary = round(INTERRUPTION_SECONDS * FPS)
    count = round((RESUME_SECONDS - INTERRUPTION_SECONDS) * FPS)
    fallback_start = nearest_idle_index(generated[boundary - 1], idle_frames)
    replacement = idle_sequence(idle_frames, fallback_start, count)

    bridge = [frame.copy() for frame in bridge[:GENERATED_BRIDGE_FRAMES]]
    target_end = nearest_idle_index(bridge[-1], idle_frames)
    target_start = (target_end - GENERATED_BRIDGE_FRAMES + 1) % len(idle_frames)
    blend_count = GENERATED_BRIDGE_FRAMES // 2
    for step in range(blend_count):
        index = GENERATED_BRIDGE_FRAMES - blend_count + step
        idle_index = (target_end - blend_count + 1 + step) % len(idle_frames)
        alpha = (step + 1) / blend_count
        bridge[index] = cv2.addWeighted(
            bridge[index], 1.0 - alpha, idle_frames[idle_index], alpha, 0
        )

    for step, frame in enumerate(bridge):
        index = generation_delay_frames + step
        if index < count:
            replacement[index] = frame
    idle_resume = (target_start + GENERATED_BRIDGE_FRAMES) % len(idle_frames)
    start = min(generation_delay_frames + len(bridge), count)
    replacement[start:] = idle_sequence(idle_frames, idle_resume, count - start)
    return replace_interruption(generated, replacement)


def hybrid_generated_bridge(
    generated: list[np.ndarray],
    idle_frames: list[np.ndarray],
    bridge: list[np.ndarray],
    generation_delay_frames: int,
) -> list[np.ndarray]:
    boundary = round(INTERRUPTION_SECONDS * FPS)
    count = round((RESUME_SECONDS - INTERRUPTION_SECONDS) * FPS)
    source = generated[boundary - 1]
    selected = [frame.copy() for frame in bridge[:HYBRID_GENERATED_FRAMES]]
    target_start = nearest_idle_index(selected[-1], idle_frames)

    replacement = [source.copy() for _ in range(count)]
    blend_count = 2
    for step in range(blend_count):
        index = len(selected) - blend_count + step
        alpha = smoothstep((step + 1) / blend_count)
        selected[index] = cv2.addWeighted(
            selected[index],
            1.0 - alpha,
            idle_frames[(target_start + step) % len(idle_frames)],
            alpha,
            0,
        )

    for step, frame in enumerate(selected):
        index = generation_delay_frames + step
        if index < count:
            replacement[index] = frame
    start = min(generation_delay_frames + len(selected), count)
    replacement[start:] = idle_sequence(
        idle_frames, target_start + blend_count, count - start
    )
    return replace_interruption(generated, replacement)


def vae_endpoint_bridge(
    runtime: Any,
    generated: list[np.ndarray],
    idle_frames: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, float]]:
    boundary = round(INTERRUPTION_SECONDS * FPS)
    count = round((RESUME_SECONDS - INTERRUPTION_SECONDS) * FPS)
    source_window = generated[boundary - MOTION_FRAMES : boundary]
    idle_start = nearest_idle_index(source_window[-1], idle_frames)
    target_window = idle_sequence(
        idle_frames, idle_start, MOTION_FRAMES + DIRECT_BRIDGE_FRAMES
    )[-MOTION_FRAMES:]

    source_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in source_window]
    target_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in target_window]
    started = time.perf_counter()
    runtime.generate_vae_transition(source_rgb, target_rgb, DIRECT_BRIDGE_FRAMES)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    bridge_rgb = runtime.generate_vae_transition(
        source_rgb, target_rgb, DIRECT_BRIDGE_FRAMES
    )
    warm_seconds = time.perf_counter() - started
    bridge = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in bridge_rgb]

    replacement = bridge + idle_sequence(
        idle_frames, idle_start + DIRECT_BRIDGE_FRAMES, count - len(bridge)
    )
    return replace_interruption(generated, replacement), {
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
    }


def label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (labeled.shape[1], 54), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.72, labeled, 0.28, 0, labeled)
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
    cv2.putText(
        labeled,
        label,
        ((labeled.shape[1] - size[0]) // 2, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def comparison_grid(strategies: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    labels = list(strategies)
    count = min(len(value) for value in strategies.values())
    blank = np.zeros_like(next(iter(strategies.values()))[0])
    output: list[np.ndarray] = []
    for index in range(count):
        tiles = [label_frame(strategies[label][index], label) for label in labels]
        while len(tiles) < 6:
            tiles.append(label_frame(blank, "INTERRUPTION AT 4.0s"))
        output.append(np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:6]))))
    return output


def interruption_metrics(frames: list[np.ndarray]) -> dict[str, float]:
    boundary = round(INTERRUPTION_SECONDS * FPS)
    end = round(RESUME_SECONDS * FPS)
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        for frame in frames[boundary - 1 : end]
    ]
    deltas = np.asarray(
        [float(np.mean(cv2.absdiff(gray[index - 1], gray[index]))) for index in range(1, len(gray))]
    )
    return {
        "first_frame_delta": float(deltas[0]),
        "mean_interruption_delta": float(np.mean(deltas)),
        "max_interruption_delta": float(np.max(deltas)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    global INTERRUPTION_SECONDS, RESUME_SECONDS, SOURCE_RESUME_SECONDS
    INTERRUPTION_SECONDS = args.interruption_seconds
    SOURCE_RESUME_SECONDS = args.source_resume_seconds
    RESUME_SECONDS = INTERRUPTION_SECONDS + args.idle_seconds
    if INTERRUPTION_SECONDS <= MOTION_FRAMES / FPS:
        raise RuntimeError("interruption must leave enough source frames for motion history")
    if SOURCE_RESUME_SECONDS <= INTERRUPTION_SECONDS:
        raise RuntimeError("source resume must occur after interruption")
    if args.idle_seconds <= 0:
        raise RuntimeError("idle duration must be positive")
    if not args.generated.is_file():
        raise RuntimeError(f"generated fixture not found: {args.generated}")
    if not args.idle_clip.is_file():
        raise RuntimeError(f"idle clip not found: {args.idle_clip}")

    sys.path.insert(0, str(ROOT_DIR))
    from flashhead_idle_strategies import decode_video, resample_loop, write_video

    from interactive_avatar.flashhead import FlashHeadConfig
    from interactive_avatar.flashhead.runtime import FlashHeadRuntime

    source_generated, generated_fps = decode_video(args.generated, target_size=(512, 512))
    if not np.isclose(generated_fps, FPS):
        raise RuntimeError(f"generated fixture must be {FPS:g} FPS, got {generated_fps:g}")
    idle_source, idle_fps = decode_video(args.idle_clip, target_size=(512, 512))
    idle_frames = resample_loop(idle_source, idle_fps, max(len(idle_source) * 25 // 24, 1))
    generated = expanded_timeline(source_generated)
    expanded_audio = args.output_dir / "interruption_options_audio.wav"
    make_expanded_audio(args.generated, expanded_audio)

    runtime = FlashHeadRuntime(FlashHeadConfig())
    print("Loading and warming FlashHead...", flush=True)
    load_seconds = runtime.load()
    warmup_seconds = runtime.warmup()
    boundary = round(INTERRUPTION_SECONDS * FPS)
    motion_rgb = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for frame in generated[boundary - MOTION_FRAMES : boundary]
    ]
    started = time.perf_counter()
    cold_bridge_rgb = runtime.generate_from_motion(
        motion_rgb, bytes(runtime.config.bytes_per_chunk)
    )
    cold_generation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm_bridge_rgb = runtime.generate_from_motion(
        motion_rgb, bytes(runtime.config.bytes_per_chunk)
    )
    warm_generation_seconds = time.perf_counter() - started
    cold_bridge = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in cold_bridge_rgb]
    warm_bridge = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in warm_bridge_rgb]
    cold_delay_frames = max(1, math.ceil(cold_generation_seconds * FPS))
    warm_delay_frames = max(1, math.ceil(warm_generation_seconds * FPS))

    hard_frames, hard_idle_start = hard_cut(generated, idle_frames)
    pixel_frames, pixel_idle_start = direct_pixel_blend(generated, idle_frames)
    current_cold_frames = current_generated_bridge(
        generated, idle_frames, cold_bridge, cold_delay_frames
    )
    current_warm_frames = current_generated_bridge(
        generated, idle_frames, warm_bridge, warm_delay_frames
    )
    hybrid_frames = hybrid_generated_bridge(
        generated, idle_frames, warm_bridge, warm_delay_frames
    )
    latent_frames, latent_timing = vae_endpoint_bridge(
        runtime, generated, idle_frames
    )

    strategies = {
        "1 CURRENT FIRST": current_cold_frames,
        "2 CURRENT WARM": current_warm_frames,
        "3 HARD CUT": hard_frames,
        "4 PIXEL BLEND 4F": pixel_frames,
        "5 GENERATED HYBRID": hybrid_frames,
        "6 VAE ENDPOINT 4F": latent_frames,
    }
    filenames = {
        "1 CURRENT FIRST": "01_current_first_interruption.mp4",
        "2 CURRENT WARM": "02_current_warm_interruption.mp4",
        "3 HARD CUT": "03_hard_cut.mp4",
        "4 PIXEL BLEND 4F": "04_pixel_blend_4f.mp4",
        "5 GENERATED HYBRID": "05_generated_hybrid.mp4",
        "6 VAE ENDPOINT 4F": "06_vae_endpoint_4f.mp4",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for name, frames in strategies.items():
        output = args.output_dir / filenames[name]
        write_video(frames, expanded_audio, output)
        outputs[name] = str(output)

    comparison = args.output_dir / "interruption_options_side_by_side.mp4"
    write_video(comparison_grid(strategies), expanded_audio, comparison)
    report = {
        "scope": f"controlled offline FlashHead interruption at {INTERRUPTION_SECONDS:.2f} seconds",
        "interruption_seconds": INTERRUPTION_SECONDS,
        "last_speaking_frame_seconds": (round(INTERRUPTION_SECONDS * FPS) - 1) / FPS,
        "idle_seconds": args.idle_seconds,
        "source_resume_seconds": SOURCE_RESUME_SECONDS,
        "generated_fixture": str(args.generated),
        "idle_clip": str(args.idle_clip),
        "flashhead_load_seconds": load_seconds,
        "flashhead_warmup_seconds": warmup_seconds,
        "first_silent_bridge_generation_seconds": cold_generation_seconds,
        "warm_silent_bridge_generation_seconds": warm_generation_seconds,
        "first_generation_delay_frames": cold_delay_frames,
        "warm_generation_delay_frames": warm_delay_frames,
        "first_generation_delay_ms": cold_delay_frames / FPS * 1000,
        "warm_generation_delay_ms": warm_delay_frames / FPS * 1000,
        "hard_cut_idle_start": hard_idle_start,
        "pixel_blend_idle_start": pixel_idle_start,
        "vae_endpoint_timing": latent_timing,
        "outputs": outputs,
        "comparison": str(comparison),
        "metrics": {name: interruption_metrics(frames) for name, frames in strategies.items()},
        "notes": {
            "current_generated": "Best-case live simulation: idle is published while GPU bridge generation completes.",
            "generated_hybrid": "Successful generated path: hold the last pose, then use four generated frames and a two-frame idle blend.",
            "vae_endpoint": "Experimental endpoint interpolation in VAE space, not endpoint-conditioned diffusion.",
        },
    }
    report_path = args.output_dir / "interruption_options_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def main() -> int:
    if not in_flashhead_environment():
        try:
            return reexec_in_flashhead_environment()
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    args = parse_args()
    try:
        report = run(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
