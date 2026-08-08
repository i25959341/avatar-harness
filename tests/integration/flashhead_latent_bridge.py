#!/usr/bin/env python3
"""Test linear and smoothstep interpolation in FlashHead's VAE latent space."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
FLASHHEAD_DIR = ROOT_DIR / "third_party" / "SoulX-FlashHead"
FLASHHEAD_PYTHON = FLASHHEAD_DIR / ".venv" / "bin" / "python"
VAE_DIR = FLASHHEAD_DIR / "models" / "SoulX-FlashHead-1_3B" / "VAE_LTX"
DEFAULT_GENERATED = (
    ROOT_DIR / "outputs" / "flashhead" / "interruption" / "flashhead_interruption_raw.mp4"
)
DEFAULT_IDLE_STRATEGIES = ROOT_DIR / "outputs" / "flashhead" / "idle_strategy_comparison"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "flashhead" / "latent_bridge"
FPS = 25.0
WINDOW_FRAMES = 9


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FlashHead VAE latent bridges.")
    parser.add_argument("--generated", type=root_path, default=DEFAULT_GENERATED)
    parser.add_argument("--idle-clip", type=root_path, default=ROOT_DIR / "assets" / "idle.mp4")
    parser.add_argument("--output-dir", type=root_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--transition-frames", type=int, default=8)
    parser.add_argument(
        "--idle-strategy-dir",
        type=root_path,
        default=DEFAULT_IDLE_STRATEGIES,
        help="directory containing the prior pixel/generated bridge comparison",
    )
    return parser.parse_args()


def in_flashhead_environment() -> bool:
    return Path(sys.prefix).resolve() == (FLASHHEAD_DIR / ".venv").resolve()


def reexec_in_flashhead_environment() -> int:
    if not FLASHHEAD_PYTHON.is_file():
        raise RuntimeError(f"FlashHead Python not found: {FLASHHEAD_PYTHON}")
    result = subprocess.run([str(FLASHHEAD_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    return result.returncode


def frame_window_ending(frames: list[np.ndarray], end_index: int) -> list[np.ndarray]:
    indices = [
        min(max(end_index - WINDOW_FRAMES + 1 + offset, 0), len(frames) - 1)
        for offset in range(WINDOW_FRAMES)
    ]
    return [frames[index] for index in indices]


def frames_to_tensor(frames: list[np.ndarray], torch: Any) -> Any:
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    tensor = torch.from_numpy(rgb).permute(3, 0, 1, 2).unsqueeze(0)
    return (tensor.to(device="cuda", dtype=torch.bfloat16) / 127.5) - 1.0


def decoded_last_frame(decoded: Any) -> np.ndarray:
    frame = decoded[0, :, -1].float().permute(1, 2, 0)
    rgb = ((frame + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def linear(value: float) -> float:
    return value


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def interpolate_bridge(
    vae: Any,
    source_window: list[np.ndarray],
    target_window: list[np.ndarray],
    transition_frames: int,
    easing: Callable[[float], float],
    torch: Any,
) -> tuple[list[np.ndarray], dict[str, float]]:
    encode_started = time.perf_counter()
    source_latent = vae.encode(frames_to_tensor(source_window, torch))
    target_latent = vae.encode(frames_to_tensor(target_window, torch))
    encode_seconds = time.perf_counter() - encode_started

    latent_distance = float(torch.mean(torch.square(target_latent - source_latent)).sqrt())
    bridge: list[np.ndarray] = []
    decode_started = time.perf_counter()
    for step in range(transition_frames):
        # The source and target pixels remain outside the bridge, so neither endpoint
        # is duplicated in the interpolated sequence.
        alpha = easing((step + 1) / (transition_frames + 1))
        latent = torch.lerp(source_latent, target_latent, alpha)
        bridge.append(decoded_last_frame(vae.decode(latent)))
    decode_seconds = time.perf_counter() - decode_started
    return bridge, {
        "latent_rms_distance": latent_distance,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
    }


def build_bridges(
    vae: Any,
    generated: list[np.ndarray],
    hard_switch: list[np.ndarray],
    transition_frames: int,
    easing: Callable[[float], float],
    torch: Any,
) -> tuple[list[np.ndarray], dict[str, dict[str, float]]]:
    from flashhead_idle_strategies import BOUNDARIES

    output = [frame.copy() for frame in hard_switch]
    timings: dict[str, dict[str, float]] = {}
    for name, timestamp, direction in BOUNDARIES:
        boundary = round(timestamp * FPS)
        if direction == "idle_to_talk":
            bridge_start = boundary - transition_frames
            source_window = frame_window_ending(hard_switch, bridge_start - 1)
            target_window = frame_window_ending(generated, boundary)
        else:
            bridge_start = boundary
            source_window = frame_window_ending(generated, boundary - 1)
            target_window = frame_window_ending(hard_switch, boundary + transition_frames)

        bridge, bridge_metrics = interpolate_bridge(
            vae,
            source_window,
            target_window,
            transition_frames,
            easing,
            torch,
        )
        output[bridge_start : bridge_start + transition_frames] = bridge
        timings[name] = bridge_metrics
    return output, timings


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.transition_frames < 2:
        raise RuntimeError("--transition-frames must be at least 2")
    if not args.generated.is_file():
        raise RuntimeError(f"generated video not found: {args.generated}")
    if not args.idle_clip.is_file():
        raise RuntimeError(f"idle clip not found: {args.idle_clip}")
    if not (VAE_DIR / "diffusion_pytorch_model.safetensors").is_file():
        raise RuntimeError(f"FlashHead VAE checkpoint not found: {VAE_DIR}")

    sys.path.insert(0, str(ROOT_DIR))
    sys.path.insert(0, str(FLASHHEAD_DIR))
    import torch
    from flash_head.ltx_video.ltx_vae import LtxVAE
    from flashhead_idle_strategies import (
        OUTPUT_SIZE,
        decode_video,
        make_comparison_grid,
        match_idle_regions,
        resample_loop,
        strategy_metrics,
        write_video,
    )

    generated, generated_fps = decode_video(args.generated, target_size=OUTPUT_SIZE)
    if not np.isclose(generated_fps, FPS):
        raise RuntimeError(f"generated video must be 25 FPS, got {generated_fps:g}")
    idle_source, idle_fps = decode_video(args.idle_clip, target_size=OUTPUT_SIZE)
    idle_loop = resample_loop(idle_source, idle_fps, len(idle_source) * 25 // 24)
    hard_switch, offsets = match_idle_regions(generated, idle_loop)

    print(f"Loading FlashHead LTX VAE from {VAE_DIR}...", flush=True)
    vae_started = time.perf_counter()
    vae = LtxVAE(VAE_DIR, dtype=torch.bfloat16, device="cuda")
    vae_load_seconds = time.perf_counter() - vae_started
    print(f"VAE loaded in {vae_load_seconds:.1f}s", flush=True)

    print("Generating linear latent bridges...", flush=True)
    linear_frames, linear_timings = build_bridges(
        vae,
        generated,
        hard_switch,
        args.transition_frames,
        linear,
        torch,
    )
    print("Generating smoothstep latent bridges...", flush=True)
    smooth_frames, smooth_timings = build_bridges(
        vae,
        generated,
        hard_switch,
        args.transition_frames,
        smoothstep,
        torch,
    )

    generated_bridge_path = args.idle_strategy_dir / "04_generated_bridge.mp4"
    crossfade_path = args.idle_strategy_dir / "03_idle_clip_crossfade.mp4"
    if not generated_bridge_path.is_file() or not crossfade_path.is_file():
        raise RuntimeError(
            "prior idle strategy outputs are missing; run "
            "tests/integration/flashhead_idle_strategies.py first"
        )
    generated_bridge, _ = decode_video(generated_bridge_path, target_size=OUTPUT_SIZE)
    crossfade, _ = decode_video(crossfade_path, target_size=OUTPUT_SIZE)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    linear_path = output_dir / "05_vae_latent_linear.mp4"
    smooth_path = output_dir / "06_vae_latent_smoothstep.mp4"
    write_video(linear_frames, args.generated, linear_path)
    write_video(smooth_frames, args.generated, smooth_path)

    strategies = {
        "PIXEL CROSSFADE": crossfade,
        "GENERATED BRIDGE": generated_bridge,
        "VAE LATENT LINEAR": linear_frames,
        "VAE LATENT SMOOTHSTEP": smooth_frames,
    }
    comparison_path = output_dir / "latent_bridge_side_by_side.mp4"
    write_video(make_comparison_grid(strategies), args.generated, comparison_path)

    metrics = {
        name: strategy_metrics(frames, args.transition_frames)
        for name, frames in strategies.items()
    }
    report: dict[str, Any] = {
        "scope": "offline FlashHead VAE interpolation; no live denoiser or LiveKit",
        "generated_source": str(args.generated),
        "idle_clip": str(args.idle_clip),
        "vae_checkpoint": str(VAE_DIR),
        "vae_load_seconds": vae_load_seconds,
        "latent_shape": [128, 2, 16, 16],
        "source_and_target_window_frames": WINDOW_FRAMES,
        "transition_frames": args.transition_frames,
        "transition_duration_ms": args.transition_frames / FPS * 1000,
        "matched_idle_offsets": offsets,
        "linear_timings": linear_timings,
        "smoothstep_timings": smooth_timings,
        "outputs": {
            "linear": str(linear_path),
            "smoothstep": str(smooth_path),
            "comparison": str(comparison_path),
        },
        "metrics": metrics,
        "warning": (
            "VAE latents contain appearance and temporal information; low frame-delta "
            "metrics do not rule out blur, morphing, or identity artifacts."
        ),
    }
    report_path = output_dir / "latent_bridge_report.json"
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
