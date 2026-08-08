import argparse
import os

import cv2
import face_alignment
import numpy as np
import torch


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"w": w, "h": h, "fps": fps}


def get_face_metrics_from_frame(frame, fa):
    """
    Extracts face size and center from a specific frame (numpy array).
    """
    # Detect landmarks (2D)
    # face_alignment expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(rgb)

    if preds is None or len(preds) == 0:
        return None

    lm = preds[0]  # (68, 2)

    x_min = np.min(lm[:, 0])
    x_max = np.max(lm[:, 0])
    y_min = np.min(lm[:, 1])
    y_max = np.max(lm[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    return {"width": width, "height": height, "center": center}


def align_videos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle", type=str, required=True, help="Path to raw idle video")
    parser.add_argument("--gen", type=str, required=True, help="Path to generated/reference video")
    parser.add_argument("--out", type=str, required=True, help="Output path")
    args = parser.parse_args()

    idle_path = args.idle
    gen_path = args.gen
    out_path = args.out

    print(f"Aligning {idle_path} to {gen_path}...")

    if not os.path.exists(idle_path) or not os.path.exists(gen_path):
        print("Error: Files not found.")
        return

    # 1. Check Resolutions
    idle_info = get_video_info(idle_path)
    gen_info = get_video_info(gen_path)

    print(f"Idle Video: {idle_info['w']}x{idle_info['h']} @ {idle_info['fps']}fps")
    print(f"Gen Video:  {gen_info['w']}x{gen_info['h']} @ {gen_info['fps']}fps")

    target_w, target_h = gen_info["w"], gen_info["h"]

    # Init Face Alignment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
    )

    # 2. Get Metrics FROM RESIZED IDLE FRAME
    # We want to compare apples to apples (512x512 space)

    # Read Gen Frame
    cap_gen = cv2.VideoCapture(gen_path)
    ret, frame_gen = cap_gen.read()
    cap_gen.release()
    metrics_gen = get_face_metrics_from_frame(frame_gen, fa)

    # Read Idle Frame & Resize
    cap_idle = cv2.VideoCapture(idle_path)
    ret, frame_idle_raw = cap_idle.read()

    # Resize Logic
    # If we just cv2.resize, we distort aspect ratio if they differ.
    # Assuming standard square/16:9 relation.
    # Safe bet: Resize straight to 512x512 for analysis
    frame_idle_resized = cv2.resize(frame_idle_raw, (target_w, target_h))
    metrics_idle = get_face_metrics_from_frame(frame_idle_resized, fa)

    if not metrics_idle or not metrics_gen:
        print("Could not detect face in one of the videos.")
        cap_idle.release()
        return

    # 3. Calculate Scale/Shift
    scale_w = metrics_gen["width"] / metrics_idle["width"]
    scale_h = metrics_gen["height"] / metrics_idle["height"]
    scale = (scale_w + scale_h) / 2.0

    print(f"Idle Face (Resized): {metrics_idle['width']:.1f}x{metrics_idle['height']:.1f}")
    print(f"Gen Face:            {metrics_gen['width']:.1f}x{metrics_gen['height']:.1f}")
    print(f"Required Scale Factor: {scale:.4f}")

    # 4. Processing Loop
    # Reset Cap
    cap_idle.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), gen_info["fps"], (target_w, target_h)
    )

    # Center Mapping
    # We apply M to the RESIZED frame.
    M = cv2.getRotationMatrix2D(tuple(metrics_idle["center"]), 0, scale)
    shift = metrics_gen["center"] - metrics_idle["center"]
    M[0, 2] += shift[0]
    M[1, 2] += shift[1]

    frame_count = 0
    while True:
        ret, frame = cap_idle.read()
        if not ret:
            break

        # Step 1: Resize to Target
        resized = cv2.resize(frame, (target_w, target_h))

        # Step 2: Warp
        # Use BORDER_REPLICATE to avoid kaleidoscope.
        # Ideally, if scale > 1, we are zooming IN, so no borders.
        # If scale < 1, we are zooming OUT, so borders appear. Replicate smears them.
        warped = cv2.warpAffine(resized, M, (target_w, target_h), borderMode=cv2.BORDER_REPLICATE)

        writer.write(warped)
        frame_count += 1

    cap_idle.release()
    writer.release()
    print(f"Processed {frame_count} frames. Saved to {out_path}")


if __name__ == "__main__":
    align_videos()
