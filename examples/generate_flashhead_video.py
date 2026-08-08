#!/usr/bin/env python3
"""Run SoulX-FlashHead Lite with the isolated environment in third_party."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FLASHHEAD_DIR = ROOT_DIR / "third_party" / "SoulX-FlashHead"


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def resolve_from_root(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{description} not found: {path}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"{description} not found: {path}")


def check_cuda(python: Path) -> str:
    script = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "print(torch.cuda.get_device_name(0))"
    )
    result = subprocess.run(
        [str(python), "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def probe_output(path: Path) -> dict[str, object] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return json.loads(result.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video with SoulX-FlashHead Lite."
    )
    parser.add_argument(
        "--source",
        type=existing_file,
        default=DEFAULT_FLASHHEAD_DIR / "examples" / "girl.png",
        help="portrait image (default: FlashHead's bundled girl.png)",
    )
    parser.add_argument(
        "--audio",
        type=existing_file,
        default=DEFAULT_FLASHHEAD_DIR / "examples" / "podcast_sichuan_16k.wav",
        help="input audio file (default: FlashHead's bundled podcast sample)",
    )
    parser.add_argument(
        "--output",
        type=resolve_from_root,
        default=ROOT_DIR / "outputs" / "flashhead" / "demo.mp4",
        help="output MP4 path (default: outputs/flashhead/demo.mp4)",
    )
    parser.add_argument("--seed", type=int, default=42, help="generation seed")
    parser.add_argument(
        "--audio-mode",
        choices=("stream", "once"),
        default="stream",
        help="audio encoding mode used by FlashHead",
    )
    parser.add_argument(
        "--face-crop",
        action="store_true",
        help="detect and crop the face before generation",
    )
    parser.add_argument(
        "--flashhead-dir",
        type=resolve_from_root,
        default=DEFAULT_FLASHHEAD_DIR,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the generation command without running it",
    )
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> Path:
    flashhead_dir: Path = args.flashhead_dir
    python = flashhead_dir / ".venv" / "bin" / "python"
    generator = flashhead_dir / "generate_video.py"
    checkpoint_dir = flashhead_dir / "models" / "SoulX-FlashHead-1_3B"
    wav2vec_dir = flashhead_dir / "models" / "wav2vec2-base-960h"
    output: Path = args.output

    require_directory(flashhead_dir, "FlashHead checkout")
    require_file(python, "FlashHead virtual-environment Python")
    require_file(generator, "FlashHead generator")
    require_file(args.source, "source image")
    require_file(args.audio, "input audio")
    require_file(
        checkpoint_dir / "Model_Lite" / "diffusion_pytorch_model.safetensors",
        "FlashHead Lite checkpoint",
    )
    require_file(
        checkpoint_dir / "VAE_LTX" / "diffusion_pytorch_model.safetensors",
        "FlashHead VAE checkpoint",
    )
    require_file(wav2vec_dir / "config.json", "Wav2Vec2 checkpoint")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(generator),
        "--ckpt_dir",
        str(checkpoint_dir),
        "--wav2vec_dir",
        str(wav2vec_dir),
        "--model_type",
        "lite",
        "--cond_image",
        str(args.source),
        "--audio_path",
        str(args.audio),
        "--audio_encode_mode",
        args.audio_mode,
        "--base_seed",
        str(args.seed),
        "--save_file",
        str(output),
    ]
    if args.face_crop:
        command.extend(["--use_face_crop", "True"])

    print("FlashHead demo")
    print(f"  source: {args.source}")
    print(f"  audio:  {args.audio}")
    print(f"  output: {output}")
    print(f"  mode:   Lite / {args.audio_mode}")

    if args.dry_run:
        print("\nCommand:")
        print(" ".join(command))
        return output

    try:
        gpu_name = check_cuda(python)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"FlashHead CUDA check failed: {detail}") from error

    print(f"  GPU:    {gpu_name}")
    print("\nLoading the model and generating video...")
    started_at = time.perf_counter()
    subprocess.run(command, cwd=flashhead_dir, check=True)
    elapsed = time.perf_counter() - started_at

    require_file(output, "generated video")
    print(f"\nFinished in {elapsed:.1f}s: {output}")
    metadata = probe_output(output)
    if metadata is not None:
        duration = float(metadata["format"]["duration"])
        size_mb = int(metadata["format"]["size"]) / (1024 * 1024)
        video = next(
            stream for stream in metadata["streams"] if stream.get("codec_type") == "video"
        )
        print(
            f"Verified {duration:.2f}s, {video.get('width')}x{video.get('height')}, "
            f"{video.get('r_frame_rate')} fps, {size_mb:.1f} MiB"
        )
    return output


def main() -> int:
    args = parse_args()
    try:
        run_demo(args)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
