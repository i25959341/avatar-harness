import os
import queue
import subprocess
import tempfile
import threading
import time
import wave

from .events import OutputFrame
from .frame_queue import FrameQueue


class StreamBroadcaster(threading.Thread):
    def __init__(
        self,
        frame_queue: FrameQueue,
        output_url: str,
        fps: int = 25,
        width: int = 512,
        height: int = 512,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.output_url = output_url
        self.fps = fps
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.stop_event = threading.Event()
        self.process: subprocess.Popen | None = None

        # Audio accumulation for file output
        self.audio_buffer = bytearray()
        self.bytes_per_frame = int(sample_rate * 2 / fps)  # 16-bit mono: 1280 bytes at 25fps
        self.silence_frame = bytes(self.bytes_per_frame)  # Pre-allocated silence

    def start_ffmpeg(self):
        # Check if outputting to file (will mux audio later) or streaming
        self.is_file_output = self.output_url.endswith(".mp4")
        self.temp_video_path = None

        if self.is_file_output:
            # Write video to temp file, will mux with audio on stop
            fd, self.temp_video_path = tempfile.mkstemp(suffix="_video.mp4")
            os.close(fd)
            output_target = self.temp_video_path
            output_format = "mp4"
        else:
            output_target = self.output_url
            output_format = "flv"

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-f",
            output_format,
            output_target,
        ]

        print(f"StreamBroadcaster: Starting FFMPEG: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def run(self):
        self.start_ffmpeg()
        print("StreamBroadcaster: Thread started.")

        frame_delay = 1.0 / self.fps
        next_frame_time = time.time()

        while not self.stop_event.is_set():
            try:
                # 1. Pace the loop (Sleep until next frame slot)
                now = time.time()
                sleep_time = max(0.0, next_frame_time - now)
                # print(f"Sleeping: {sleep_time:.4f}")
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Advance target for next cycle
                next_frame_time += frame_delay
                # If we are WAY behind (e.g. paused for 5s), reset timeline to avoid burst
                if next_frame_time < time.time() - 0.5:
                    next_frame_time = time.time()

                # 2. Get Frame
                # Non-blocking or short timeout since we already slept
                frame: OutputFrame = self.frame_queue.get(timeout=0.01)

                if frame is None:
                    continue

                # Write video bytes
                if self.process and self.process.stdin:
                    try:
                        self.process.stdin.write(frame.video_frame.tobytes())
                    except BrokenPipeError:
                        print("StreamBroadcaster: FFMPEG broken pipe. Stopping.")
                        self.stop_event.set()
                        break

                # Accumulate audio for file output (silence if None)
                if self.is_file_output:
                    if frame.audio_frame is not None:
                        self.audio_buffer.extend(frame.audio_frame)
                    else:
                        self.audio_buffer.extend(self.silence_frame)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"StreamBroadcaster: Unexpected error: {e}")
                self.stop_event.set()
                break

        # Cleanup
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait()

        # Mux audio with video for file output
        if self.is_file_output:
            self._mux_audio_video()

        print("StreamBroadcaster: Thread stopped.")

    def _mux_audio_video(self):
        """Combine accumulated audio with video file."""
        if len(self.audio_buffer) == 0:
            # No audio, just rename temp to final
            if self.temp_video_path and os.path.exists(self.temp_video_path):
                os.rename(self.temp_video_path, self.output_url)
                print(f"StreamBroadcaster: Output saved (no audio): {self.output_url}")
            return

        print("StreamBroadcaster: Muxing audio with video...")

        # Write audio buffer to temp WAV file
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            with wave.open(temp_audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(bytes(self.audio_buffer))

            # Mux video + audio
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                self.temp_video_path,
                "-i",
                temp_audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                self.output_url,
            ]

            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if result.returncode == 0:
                print(f"StreamBroadcaster: Output saved with audio: {self.output_url}")
            else:
                # Mux failed, keep video only
                print("StreamBroadcaster: Mux failed, saving video only")
                os.rename(self.temp_video_path, self.output_url)
                self.temp_video_path = None  # Prevent cleanup from deleting

        finally:
            # Cleanup temp files
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if self.temp_video_path and os.path.exists(self.temp_video_path):
                os.remove(self.temp_video_path)

    def stop(self):
        self.stop_event.set()
        self.join()
