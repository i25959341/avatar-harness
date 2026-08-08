#!/usr/bin/env python3
"""Benchmark the TaekyungKi interactive AvatarForcing release."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT_DIR / "third_party" / "InteractiveAvatarForcing"
UPSTREAM_PYTHON = UPSTREAM_DIR / ".venv" / "bin" / "python"
COMPAT_DIR = ROOT_DIR / "interactive_avatar" / "interactive_avatarforcing_compat"

if Path(sys.prefix).resolve() != UPSTREAM_PYTHON.parent.parent.resolve():
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(COMPAT_DIR),
            str(UPSTREAM_DIR),
            str(ROOT_DIR),
            environment.get("PYTHONPATH", ""),
        ]
    )
    os.execve(
        str(UPSTREAM_PYTHON),
        [str(UPSTREAM_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

sys.path.insert(0, str(UPSTREAM_DIR))

import inference as upstream_inference  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--avatar",
        type=root_path,
        default=UPSTREAM_DIR / "data" / "rumi.jpg",
    )
    parser.add_argument(
        "--avatar-audio",
        type=root_path,
        default=UPSTREAM_DIR / "data" / "avatar.wav",
    )
    parser.add_argument(
        "--user-audio",
        type=root_path,
        default=ROOT_DIR
        / "outputs"
        / "interactive-avatarforcing"
        / "input"
        / "user.wav",
    )
    parser.add_argument(
        "--user-video",
        type=root_path,
        default=ROOT_DIR
        / "outputs"
        / "interactive-avatarforcing"
        / "input"
        / "user",
    )
    parser.add_argument(
        "--output-dir",
        type=root_path,
        default=ROOT_DIR / "outputs" / "interactive-avatarforcing" / "benchmark",
    )
    parser.add_argument("--nfe", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--seed", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing_inputs = [
        path
        for path in (args.avatar, args.avatar_audio, args.user_audio, args.user_video)
        if not path.exists()
    ]
    if missing_inputs:
        raise RuntimeError(
            "missing benchmark input: "
            + ", ".join(str(path) for path in missing_inputs)
            + "; run `python3 tools/setup_interactive_avatarforcing.py` first"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream_inference.seed_everything(args.seed)

    config = OmegaConf.load(UPSTREAM_DIR / "configs" / "inference.yaml")
    config.rank = 0
    config.ngpus = 1
    config.result_dir = str(args.output_dir)
    config.mae_ckpt_path = str(UPSTREAM_DIR / "pretrained_dir" / "motion_autoencoder.pth")
    config.ckpt_path = str(UPSTREAM_DIR / "pretrained_dir" / "flow_transformer.pth")
    config.wav2vec_model_path = str(
        UPSTREAM_DIR / "pretrained_dir" / "wav2vec2-base-960h"
    )

    load_started = time.perf_counter()
    # Upstream init_network reads its CLI's module-global `opt` instead of self.opt.
    upstream_inference.opt = config
    agent = upstream_inference.InferenceAgent(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    preprocess_started = time.perf_counter()
    data = agent.data_processor.preprocess(
        avatar_ref_path=str(args.avatar),
        avatar_audio_path=str(args.avatar_audio),
        user_audio_path=str(args.user_audio),
        user_video_path=str(args.user_video),
    )
    preprocess_seconds = time.perf_counter() - preprocess_started
    media_seconds = max(data["avatar_a"].shape[-1], data["user_a"].shape[-1]) / 16000

    runs = []
    for nfe in args.nfe:
        upstream_inference.seed_everything(args.seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            frames = agent.G.inference(
                data=dict(data),
                a_cfg_scale=2.0,
                u_cfg_scale=1.0,
                nfe=nfe,
                seed=args.seed,
                use_kv_cache=True,
            )["d_hat"]
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started

        output = args.output_dir / f"official_sample_nfe{nfe}.mp4"
        agent.save_video(frames, str(output), str(args.avatar_audio))
        runs.append(
            {
                "nfe": nfe,
                "output": str(output.relative_to(ROOT_DIR)),
                "media_seconds": media_seconds,
                "generation_seconds": generation_seconds,
                "realtime_factor": media_seconds / generation_seconds,
                "peak_cuda_gib": torch.cuda.max_memory_allocated() / (1024**3),
            }
        )
        print(json.dumps(runs[-1], indent=2), flush=True)

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "load_seconds": load_seconds,
        "preprocess_seconds": preprocess_seconds,
        "runs": runs,
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
