#!/usr/bin/env python3
"""Render a persistent AVTR-1 idle/listen/speak/idle timeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
AVTR1_DIR = ROOT_DIR / "third_party" / "avtr-1"
AVTR1_PYTHON = AVTR1_DIR / ".pixi" / "envs" / "renderer" / "bin" / "python"

if Path(sys.prefix).resolve() != AVTR1_PYTHON.parent.parent.resolve():
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(AVTR1_DIR / "src"), str(ROOT_DIR)]
    )
    environment.setdefault("AVTR1_LOCAL_STORAGE", str(AVTR1_DIR / "artifacts"))
    os.execve(
        str(AVTR1_PYTHON),
        [str(AVTR1_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402
import torch  # noqa: E402
from avtr1_renderer.pipeline import Pipeline  # noqa: E402
from avtr1_renderer.types import Chunk, RenderOptions  # noqa: E402

SAMPLE_RATE = 16_000
FPS = 25


def root_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, SAMPLE_RATE, quality="HQ")
    return mono.astype(np.float32)


def fit_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    repeats = max(1, (samples + len(audio) - 1) // len(audio))
    return np.tile(audio, repeats)[:samples]


def add_label(frame_i420: np.ndarray, label: str, width: int, height: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_i420.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_I420)
    cv2.rectangle(rgb, (18, 18), (285, 62), (0, 0, 0), -1)
    cv2.putText(
        rgb,
        label,
        (30, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speech",
        type=root_path,
        default=AVTR1_DIR / "example" / "speaker_1.ogg",
    )
    parser.add_argument(
        "--listen",
        type=root_path,
        default=AVTR1_DIR / "example" / "speaker_2.ogg",
    )
    parser.add_argument("--avatar", default="maria")
    parser.add_argument("--background", default="plain_white")
    parser.add_argument(
        "--output",
        type=root_path,
        default=ROOT_DIR / "outputs" / "avtr1" / "continuous_review.mp4",
    )
    args = parser.parse_args()

    phase_chunks = [
        ("IDLE BEFORE", 10),
        ("LISTENING", 20),
        ("SPEAKING", 20),
        ("IDLE AFTER", 20),
    ]
    step_samples = 5 * SAMPLE_RATE // FPS
    total_samples = sum(count for _, count in phase_chunks) * step_samples
    speech = np.zeros(total_samples, dtype=np.float32)
    listen = np.zeros(total_samples, dtype=np.float32)
    phase_by_chunk: list[str] = []
    offset = 0
    for phase, count in phase_chunks:
        samples = count * step_samples
        if phase == "SPEAKING":
            speech[offset : offset + samples] = fit_audio(load_audio(args.speech), samples)
        elif phase == "LISTENING":
            listen[offset : offset + samples] = fit_audio(load_audio(args.listen), samples)
        phase_by_chunk.extend([phase] * count)
        offset += samples

    torch.manual_seed(42)
    np.random.seed(42)
    load_started = time.perf_counter()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[args.avatar])
    load_seconds = time.perf_counter() - load_started
    avatar = registry[args.avatar]
    motion = pipeline._motion_generator
    window_samples = (motion.chunk_size + motion.future_size) * motion.frame_len + motion.audio_shift
    out_height, out_width = avatar.source.shape[-2:]
    options = RenderOptions(pixel_format="yuv_i420", bg_id=args.background)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audio_path = args.output.with_suffix(".speech.wav")
    sf.write(audio_path, speech, SAMPLE_RATE, subtype="PCM_16")
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        size=(out_width, out_height),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="yuv420p",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=1,
        audio_path=str(audio_path),
        audio_codec="aac",
    )
    writer.send(None)

    state = None
    timings: list[dict[str, float | int | str]] = []
    try:
        for index, phase in enumerate(phase_by_chunk):
            start = index * step_samples
            speech_window = speech[start : start + window_samples]
            listen_window = listen[start : start + window_samples]
            speech_window = np.pad(speech_window, (0, window_samples - len(speech_window)))
            listen_window = np.pad(listen_window, (0, window_samples - len(listen_window)))
            started = time.perf_counter()
            state, frames = pipeline.process_chunk(
                avatar,
                Chunk(speech_window.astype(np.float32), listen_window.astype(np.float32)),
                state,
                options,
            )
            for frame in frames:
                labeled = add_label(frame.data, phase, out_width, out_height)
                writer.send(labeled.tobytes())
            elapsed = time.perf_counter() - started
            timings.append({"index": index, "phase": phase, "seconds": elapsed})
    finally:
        writer.close()
        audio_path.unlink(missing_ok=True)

    phase_metrics = {}
    for phase, _ in phase_chunks:
        values = [float(item["seconds"]) for item in timings if item["phase"] == phase]
        phase_metrics[phase] = {
            "mean_ms": float(np.mean(values) * 1000),
            "max_ms": float(np.max(values) * 1000),
            "realtime_factor": 0.2 / float(np.mean(values)),
        }
    report = {
        "output": str(args.output),
        "load_seconds": load_seconds,
        "duration_seconds": len(phase_by_chunk) * 0.2,
        "resolution": [out_width, out_height],
        "phase_metrics": phase_metrics,
        "per_chunk": timings,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
