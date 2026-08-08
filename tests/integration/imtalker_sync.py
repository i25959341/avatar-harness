#!/usr/bin/env python3
"""
Test script to simulate LiveKit audio flow and check A/V sync.

Simulates:
1. Audio arriving from TTS in chunks (like QueueAudioOutput)
2. Producer generating video+audio frames
3. Output video with audio to verify sync

Usage:
    python tests/integration/imtalker_sync.py --audio path/to/audio.wav
"""

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "avatar_models" / "imtalker"))

import cv2
import torch
from PIL import Image

from interactive_avatar.adaptive_chunker import AdaptiveChunkConfig
from interactive_avatar.adaptive_producer import AdaptiveFrameGenerator
from interactive_avatar.events import FrameType
from interactive_avatar.frame_queue import FrameQueue
from interactive_avatar.idle_injector import IdleFramePusher


def load_audio(audio_path: str) -> bytes:
    """Load audio file and return as 16kHz PCM bytes."""
    import librosa

    print(f"Loading audio: {audio_path}")
    speech_array, sr = librosa.load(audio_path, sr=16000)
    speech_pcm = (speech_array * 32767).astype(np.int16).tobytes()
    duration = len(speech_pcm) / 2 / 16000
    print(f"Audio loaded: {len(speech_pcm)} bytes, {duration:.2f}s")
    return speech_pcm


class SyncTester:
    """Test harness for A/V sync."""

    def __init__(
        self,
        source_image_path: str,
        idle_cache_path: str,
        device: str = "cuda",
        min_chunk_frames: int = 10,
    ):
        self.device = device
        self.source_image_path = source_image_path
        self.idle_cache_path = idle_cache_path
        self.min_chunk_frames = min_chunk_frames

        self.agent = None
        self.frame_queue = None
        self.audio_queue = None
        self.producer = None
        self.idle_pusher = None

    def initialize(self):
        """Initialize model and components."""
        print("Initializing model...")
        from app import AppConfig, InferenceAgent

        opt = AppConfig()
        opt.device = self.device
        self.agent = InferenceAgent(opt)

        # Pre-process source identity
        img_pil = Image.open(self.source_image_path).convert("RGB")
        s_pil = self.agent.data_processor.process_img(img_pil)
        s_tensor = self.agent.data_processor.transform(s_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            f_r, g_r = self.agent.renderer.dense_feature_encoder(s_tensor)
            t_lat = self.agent.renderer.latent_token_encoder(s_tensor)
            if isinstance(t_lat, tuple):
                t_lat = t_lat[0]
            ta_r = self.agent.renderer.adapt(t_lat, g_r)
            m_r = self.agent.renderer.latent_token_decoder(ta_r)

        source_features = {"f_r": f_r, "g_r": g_r, "t_lat": t_lat, "m_r": m_r}

        self.frame_queue = FrameQueue(max_size=200, history_size=20)
        self.audio_queue = queue.Queue()

        print(f"Using min_chunk_frames={self.min_chunk_frames} ({self.min_chunk_frames * 40}ms)")

        chunk_config = AdaptiveChunkConfig(
            min_chunk_frames=self.min_chunk_frames,
            max_chunk_frames=50,
            default_chunk_frames=max(self.min_chunk_frames, 25),
        )

        self.idle_pusher = IdleFramePusher(
            frame_queue=self.frame_queue,
            idle_cache_path=self.idle_cache_path,
            fps=25,
            agent=self.agent,
            source_features=source_features,
        )

        self.producer = AdaptiveFrameGenerator(
            agent=self.agent,
            frame_queue=self.frame_queue,
            source_features=source_features,
            input_audio_queue=self.audio_queue,
            opt=opt,
            idle_pusher=self.idle_pusher,
            config=chunk_config,
        )

        print("Model initialized.")

    def start(self):
        print("Starting threads...")
        self.idle_pusher.start()
        self.producer.start()
        print("Threads started.")

    def stop(self):
        print("Stopping...")
        if self.producer:
            self.producer.stop_event.set()
            self.producer.join(timeout=2.0)
        if self.idle_pusher:
            self.idle_pusher.stop_event.set()
            self.idle_pusher.join(timeout=2.0)
        print("Stopped.")


def stream_audio(tester: SyncTester, audio_pcm: bytes, chunk_ms: float = 20):
    """Stream audio in real-time chunks like LiveKit TTS."""
    bytes_per_ms = 16000 * 2 / 1000  # 32 bytes/ms at 16kHz 16-bit
    chunk_size = int(bytes_per_ms * chunk_ms)

    print(
        f"Streaming audio: {len(audio_pcm)} bytes in {chunk_size}-byte chunks ({chunk_ms}ms each)"
    )

    for i in range(0, len(audio_pcm), chunk_size):
        chunk = audio_pcm[i : i + chunk_size]
        tester.audio_queue.put(chunk)
        time.sleep(chunk_ms / 1000)

    print("Audio stream complete, signaling end...")
    tester.producer.end_audio_stream()


def run_test(tester: SyncTester, audio_pcm: bytes, output_path: str):
    """
    Run the test: idle -> speak -> idle, save video with audio.
    """
    FPS = 25
    FRAME_INTERVAL = 1.0 / FPS  # 40ms

    audio_duration = len(audio_pcm) / 2 / 16000
    idle_before = 1.0  # 1s idle before speaking
    idle_after = 1.0  # 1s idle after speaking
    total_duration = idle_before + audio_duration + idle_after + 1.0  # extra buffer

    print("\n=== Test Plan ===")
    print(f"  Idle: {idle_before}s")
    print(f"  Speaking: {audio_duration:.2f}s")
    print(f"  Idle after: {idle_after}s")
    print(f"  Total: ~{total_duration:.1f}s")
    print()

    # Collect frames at 25fps rate
    collected_video = []
    collected_audio = []

    # Start audio streaming after idle period (in background thread)
    def delayed_audio_stream():
        time.sleep(idle_before)
        print(">>> Starting audio stream")
        stream_audio(tester, audio_pcm, chunk_ms=20)

    audio_thread = threading.Thread(target=delayed_audio_stream)
    audio_thread.start()

    # Consume frames at real-time rate
    print("Consuming frames at 25fps...")
    start_time = time.time()
    next_frame_time = start_time
    frame_count = 0
    speaking_count = 0

    while time.time() - start_time < total_duration:
        # Wait until next frame time
        now = time.time()
        if now < next_frame_time:
            time.sleep(next_frame_time - now)

        # Get frame from queue
        try:
            frame = tester.frame_queue.get(timeout=0.1)
            if frame is None:
                next_frame_time += FRAME_INTERVAL
                continue
        except queue.Empty:
            next_frame_time += FRAME_INTERVAL
            continue

        frame_count += 1

        # Collect video (RGB)
        collected_video.append(frame.video_frame)

        # Collect audio (PCM bytes or silence)
        if frame.audio_frame is not None and len(frame.audio_frame) > 0:
            collected_audio.append(frame.audio_frame)
        else:
            # Silence: 640 bytes = 20ms at 16kHz 16-bit
            silence = bytes(640)
            collected_audio.append(silence)

        if frame.type == FrameType.SPEAKING:
            speaking_count += 1
            if speaking_count == 1:
                print(f">>> First speaking frame at {time.time() - start_time:.2f}s")

        if frame.final_chunk:
            print(f">>> Final chunk at {time.time() - start_time:.2f}s")

        next_frame_time += FRAME_INTERVAL

    audio_thread.join()

    print(f"\nCollected {frame_count} frames ({speaking_count} speaking)")

    # Write video with audio using ffmpeg
    write_video_with_audio(collected_video, collected_audio, output_path, FPS)


def write_video_with_audio(video_frames, audio_chunks, output_path: str, fps: int):
    """Write video frames and audio to output file."""
    import subprocess
    import tempfile

    if not video_frames:
        print("No frames to write!")
        return

    print(f"Writing {len(video_frames)} frames to {output_path}...")

    # Write raw video to temp file
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as vf:
        video_temp = vf.name
        for frame in video_frames:
            # Convert RGB to BGR and write
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vf.write(bgr.tobytes())

    # Write raw audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as af:
        audio_temp = af.name
        for chunk in audio_chunks:
            af.write(chunk)

    # Get frame dimensions
    h, w = video_frames[0].shape[:2]

    # Use ffmpeg to combine
    cmd = [
        "ffmpeg",
        "-y",
        # Video input
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        video_temp,
        # Audio input
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-i",
        audio_temp,
        # Output
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
    finally:
        os.unlink(video_temp)
        os.unlink(audio_temp)


def main():
    parser = argparse.ArgumentParser(description="Test A/V sync")
    parser.add_argument(
        "--source", type=str, default=str(ROOT_DIR / "avatar_models/imtalker/assets/source_1.png")
    )
    parser.add_argument(
        "--cache", type=str, default=str(ROOT_DIR / "outputs/cache/imtalker_idle.pt")
    )
    parser.add_argument(
        "--audio", type=str, default=str(ROOT_DIR / "avatar_models/imtalker/assets/audio_1.wav")
    )
    parser.add_argument("--output", type=str, default=str(ROOT_DIR / "outputs/imtalker/sync.mp4"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--min-frames", type=int, default=10, help="Minimum chunk frames (default 10 = 400ms)"
    )
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    audio_pcm = load_audio(args.audio)

    tester = SyncTester(
        source_image_path=args.source,
        idle_cache_path=args.cache,
        device=args.device,
        min_chunk_frames=args.min_frames,
    )

    try:
        tester.initialize()
        tester.start()

        # Wait for idle frames to populate
        print("Waiting for idle frames to start...")
        time.sleep(0.5)

        run_test(tester, audio_pcm, args.output)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tester.stop()

    print("\nDone! Check", args.output, "for A/V sync")


if __name__ == "__main__":
    main()
