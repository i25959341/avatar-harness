#!/usr/bin/env python3
"""
Latency test for TalkBox adaptive chunking.

Tests different chunk configurations to measure time-to-first-frame (TTFF)
with realistic streaming audio input.
"""

import argparse
import queue
import sys
import threading
import time
import wave
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "avatar_models" / "imtalker"))

import torch
from app import AppConfig, InferenceAgent
from PIL import Image

from interactive_avatar.adaptive_chunker import AdaptiveChunkConfig
from interactive_avatar.adaptive_producer import AdaptiveFrameGenerator
from interactive_avatar.events import FrameType
from interactive_avatar.frame_queue import FrameQueue


def stream_audio_realtime(audio_queue: queue.Queue, audio_bytes: bytes, fps: int = 25):
    """Stream audio at real-time rate, simulating TTS output."""
    bytes_per_frame = 1280  # 16kHz * 2 bytes / 25fps
    frame_interval = 1.0 / fps

    position = 0
    while position < len(audio_bytes):
        chunk = audio_bytes[position : position + bytes_per_frame]
        if chunk:
            audio_queue.put(chunk)
        position += bytes_per_frame
        time.sleep(frame_interval)


def measure_ttff(
    agent,
    source_features,
    opt,
    audio_bytes: bytes,
    config: AdaptiveChunkConfig,
    timeout: float = 15.0,
) -> dict:
    """Measure TTFF with streaming audio."""
    results = {
        "ttff_ms": None,
        "min_chunk_ms": config.min_chunk_frames * 40,
        "speaking_frames": 0,
    }

    frame_queue = FrameQueue(max_size=200, history_size=20)
    audio_queue = queue.Queue()

    producer = AdaptiveFrameGenerator(
        agent=agent,
        frame_queue=frame_queue,
        source_features=source_features,
        input_audio_queue=audio_queue,
        opt=opt,
        idle_pusher=None,
        config=config,
    )

    producer.start()

    # Stream audio in background
    start_time = time.perf_counter()
    stream_thread = threading.Thread(
        target=stream_audio_realtime, args=(audio_queue, audio_bytes, 25)
    )
    stream_thread.start()

    # Wait for first speaking frame
    first_frame_time = None
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            frame = frame_queue.get(timeout=0.02)
            if frame is not None and frame.type == FrameType.SPEAKING:
                if first_frame_time is None:
                    first_frame_time = time.perf_counter()
                    results["ttff_ms"] = (first_frame_time - start_time) * 1000
                results["speaking_frames"] += 1
                if results["speaking_frames"] >= 25:
                    break
        except queue.Empty:
            pass

    producer.stop_event.set()
    stream_thread.join(timeout=1.0)
    producer.join(timeout=2.0)

    return results


def main():
    parser = argparse.ArgumentParser(description="TalkBox latency test")
    parser.add_argument("--source", type=str, required=True, help="Source image")
    parser.add_argument("--audio", type=str, required=True, help="Test audio file")
    args = parser.parse_args()

    # Load audio
    print(f"Loading audio: {args.audio}")
    with wave.open(args.audio, "rb") as wf:
        audio_bytes = wf.readframes(wf.getnframes())

    test_audio = audio_bytes[:160000]  # 5 seconds
    print(f"Test audio: {len(test_audio)} bytes ({len(test_audio) / 32000:.2f}s)")

    # Initialize model
    print("\nLoading IMTalker model...")
    opt = AppConfig()
    opt.device = "cuda"
    agent = InferenceAgent(opt)

    # Pre-process source
    print(f"Processing source: {args.source}")
    img_pil = Image.open(args.source).convert("RGB")
    s_pil = agent.data_processor.process_img(img_pil)
    s_tensor = agent.data_processor.transform(s_pil).unsqueeze(0).to(agent.device)

    with torch.no_grad():
        f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
        t_lat = agent.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        ta_r = agent.renderer.adapt(t_lat, g_r)
        m_r = agent.renderer.latent_token_decoder(ta_r)

    source_features = {"f_r": f_r, "g_r": g_r, "t_lat": t_lat, "m_r": m_r}

    print("\n" + "=" * 60)
    print("LATENCY TEST (Streaming Audio)")
    print("=" * 60)

    configs = [
        ("10 frames (400ms)", AdaptiveChunkConfig(min_chunk_frames=10)),
        ("15 frames (600ms)", AdaptiveChunkConfig(min_chunk_frames=15)),
        ("25 frames (1000ms)", AdaptiveChunkConfig(min_chunk_frames=25)),
    ]

    results_all = []

    for name, config in configs:
        print(f"\n>>> {name}")

        # Run 2 times and average
        ttffs = []
        for i in range(2):
            result = measure_ttff(agent, source_features, opt, test_audio, config)
            if result["ttff_ms"]:
                ttffs.append(result["ttff_ms"])
                print(f"    Run {i + 1}: TTFF={result['ttff_ms']:.0f}ms")

        if ttffs:
            avg = sum(ttffs) / len(ttffs)
            results_all.append((name, avg, result["min_chunk_ms"]))
            print(f"    Average: {avg:.0f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  {'Config':<25} {'Audio Wait':<12} {'TTFF':<10}")
    print(f"  {'-' * 47}")
    for name, ttff, wait in results_all:
        print(f"  {name:<25} {wait:<12}ms {ttff:<10.0f}ms")


if __name__ == "__main__":
    main()
