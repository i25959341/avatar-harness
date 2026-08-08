#!/usr/bin/env python3
"""
Generate idle frame cache from source image + driving idle video.

Uses IMTalker's video-driven mode to transfer idle motion from
a generic driving video onto the source avatar image.

Usage:
    python tools/generate_idle_cache.py \
        --source avatar_models/imtalker/assets/source_1.png \
        --driver assets/imtalker_idle_driver.mp4 \
        --output outputs/cache/imtalker_idle.pt
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "avatar_models" / "imtalker"))

from app import AppConfig, InferenceAgent

from interactive_avatar.events import FrameType, OutputFrame


def generate_idle_cache(
    source_image_path: str,
    driving_video_path: str,
    output_path: str,
    device: str = "cuda",
    crop_source: bool = True,
    crop_driver: bool = False,
    preview_video: str = None,
):
    """
    Generate idle frame cache using video-driven motion transfer.

    Args:
        source_image_path: Path to avatar source image
        driving_video_path: Path to idle driving video
        output_path: Where to save the .pt cache file
        device: cuda or cpu
        crop_source: Whether to auto-crop source image face
        crop_driver: Whether to auto-crop driving video
    """
    Path(output_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    if preview_video:
        Path(preview_video).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Idle Cache Generator")
    print("=" * 60)
    print(f"Source: {source_image_path}")
    print(f"Driver: {driving_video_path}")
    print(f"Output: {output_path}")
    print()

    # 1. Load model
    print("Loading IMTalker model...")
    opt = AppConfig()
    opt.device = device
    agent = InferenceAgent(opt)

    # 2. Process source image
    print("Processing source image...")
    img_pil = Image.open(source_image_path).convert("RGB")
    if crop_source:
        s_pil = agent.data_processor.process_img(img_pil)
    else:
        s_pil = img_pil.resize((opt.input_size, opt.input_size))

    s_tensor = agent.data_processor.transform(s_pil).unsqueeze(0).to(device)

    # 3. Extract source identity features (same as engine.py)
    print("Extracting identity features...")
    with torch.no_grad():
        f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
        t_lat = agent.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]

        # Reference motion for decoding
        ta_r = agent.renderer.adapt(t_lat, g_r)
        m_r = agent.renderer.latent_token_decoder(ta_r)

    # 4. Open driving video
    print("Processing driving video...")
    cap = cv2.VideoCapture(driving_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {driving_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}")

    # 5. Generate idle frames
    idle_frames = []

    # Calculate silent audio chunk size (16kHz, 16-bit, mono)
    bytes_per_frame = int(16000 * 2 / fps)  # ~1280 bytes at 25fps
    silent_audio = bytes(bytes_per_frame)

    print("Generating idle frames...")
    with torch.no_grad():
        for _frame_idx in tqdm(range(total_frames), desc="Processing"):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert and resize driving frame
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).resize((opt.input_size, opt.input_size))
            d_tensor = agent.data_processor.transform(frame_pil).unsqueeze(0).to(device)

            # Extract motion from driving frame
            t_c = agent.renderer.latent_token_encoder(d_tensor)
            if isinstance(t_c, tuple):
                t_c = t_c[0]

            # Apply to source identity
            ta_c = agent.renderer.adapt(t_c, g_r)
            m_c = agent.renderer.latent_token_decoder(ta_c)

            # Decode to frame
            out_frame = agent.renderer.decode(m_c, m_r, f_r)

            # Convert to numpy uint8
            frame_np = out_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)

            # Create OutputFrame
            output_frame = OutputFrame(
                video_frame=frame_np,
                audio_frame=silent_audio,
                motion_latent=t_c.cpu().clone(),
                type=FrameType.IDLE,
            )

            idle_frames.append(output_frame)

    cap.release()

    # 6. Save cache
    print(f"\nSaving {len(idle_frames)} frames to {output_path}...")
    torch.save(idle_frames, output_path)

    # 7. Generate preview video if requested
    if preview_video:
        print(f"Generating preview video: {preview_video}")
        _save_preview_video(idle_frames, preview_video, fps)

    # 8. Summary
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    duration_sec = len(idle_frames) / fps

    print()
    print("=" * 60)
    print("Done!")
    print(f"  Frames: {len(idle_frames)}")
    print(f"  Duration: {duration_sec:.1f}s")
    print(f"  Cache size: {file_size_mb:.1f} MB")
    if preview_video:
        print(f"  Preview: {preview_video}")
    print("=" * 60)


def _save_preview_video(frames: list, output_path: str, fps: float):
    """Save frames as MP4 video for preview."""
    import subprocess

    # Use ffmpeg to encode
    height, width = frames[0].video_frame.shape[:2]

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    for frame in frames:
        process.stdin.write(frame.video_frame.tobytes())

    process.stdin.close()
    process.wait()


def main():
    parser = argparse.ArgumentParser(description="Generate idle frame cache")
    parser.add_argument(
        "--source",
        type=str,
        default=str(ROOT_DIR / "avatar_models/imtalker/assets/source_1.png"),
        help="Path to source avatar image",
    )
    parser.add_argument(
        "--driver",
        type=str,
        default=str(ROOT_DIR / "assets/imtalker_idle_driver.mp4"),
        help="Path to driving idle video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "outputs/cache/imtalker_idle.pt"),
        help="Output cache file path",
    )
    parser.add_argument(
        "--preview",
        type=str,
        default=None,
        help="Also save preview video (e.g., outputs/imtalker/idle_preview.mp4)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--no-crop-source", action="store_true", help="Don't auto-crop source image"
    )
    parser.add_argument("--crop-driver", action="store_true", help="Auto-crop driving video frames")

    args = parser.parse_args()

    generate_idle_cache(
        source_image_path=args.source,
        driving_video_path=args.driver,
        output_path=args.output,
        device=args.device,
        crop_source=not args.no_crop_source,
        crop_driver=args.crop_driver,
        preview_video=args.preview,
    )


if __name__ == "__main__":
    main()
