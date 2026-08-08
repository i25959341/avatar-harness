import queue

import torch
from app import AppConfig, InferenceAgent
from PIL import Image

from .adaptive_chunker import AdaptiveChunkConfig
from .adaptive_producer import AdaptiveFrameGenerator
from .consumer import StreamBroadcaster
from .frame_queue import FrameQueue
from .idle_injector import IdleFramePusher


class IMTalkerEngine:
    def __init__(
        self,
        source_image_path: str,
        idle_cache_path: str,
        output_url: str = "rtmp://localhost:1935/live/stream",
        device: str = "cuda",
        chunk_config: AdaptiveChunkConfig | None = None,
    ):

        print("IMTalkerEngine: Initializing...")

        self.chunk_config = chunk_config or AdaptiveChunkConfig()

        # 1. Config & Model Loading
        self.opt = AppConfig()
        self.opt.device = device
        self.agent = InferenceAgent(self.opt)

        # 2. Pre-process Source Identity
        print(f"IMTalkerEngine: Caching identity from {source_image_path}...")
        img_pil = Image.open(source_image_path).convert("RGB")
        s_pil = self.agent.data_processor.process_img(img_pil)
        s_tensor = self.agent.data_processor.transform(s_pil).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            f_r, g_r = self.agent.renderer.dense_feature_encoder(s_tensor)
            t_lat = self.agent.renderer.latent_token_encoder(s_tensor)
            if isinstance(t_lat, tuple):
                t_lat = t_lat[0]

            # Pre-decode reference motion for decoding
            ta_r = self.agent.renderer.adapt(t_lat, g_r)
            m_r = self.agent.renderer.latent_token_decoder(ta_r)

        # Bundle features for Producer
        self.source_features = {"f_r": f_r, "g_r": g_r, "t_lat": t_lat, "m_r": m_r}

        # 3. Components
        self.frame_queue = FrameQueue(max_size=200, history_size=20)
        self.audio_input_queue = queue.Queue()

        # Consumer (Broadcaster)
        self.broadcaster = StreamBroadcaster(
            frame_queue=self.frame_queue, output_url=output_url, fps=self.opt.fps
        )

        # Idle Injector (with renderer for latent interpolation)
        self.idle_pusher = IdleFramePusher(
            frame_queue=self.frame_queue,
            idle_cache_path=idle_cache_path,
            fps=self.opt.fps,
            agent=self.agent,
            source_features=self.source_features,
        )

        # Producer (Adaptive Frame Generator)
        print(
            f"IMTalkerEngine: Chunk config - "
            f"min={self.chunk_config.min_chunk_frames} frames ({self.chunk_config.min_chunk_frames * 40}ms), "
            f"max={self.chunk_config.max_chunk_frames} frames ({self.chunk_config.max_chunk_frames * 40}ms)"
        )
        self.producer = AdaptiveFrameGenerator(
            agent=self.agent,
            frame_queue=self.frame_queue,
            source_features=self.source_features,
            input_audio_queue=self.audio_input_queue,
            opt=self.opt,
            idle_pusher=self.idle_pusher,
            config=self.chunk_config,
        )

        print("IMTalkerEngine: Ready.")

    def start(self):
        print("IMTalkerEngine: Starting threads...")
        self.broadcaster.start()
        self.idle_pusher.start()
        self.producer.start()

        # Start the logic loop (Transition Manager)
        # For now, simplistic logic:
        # If we pushed audio, we tell IdlePusher to back off?
        # Actually, FrameGeneratorWorker consumes audio.
        # We need a manager to toggle self.idle_pusher.set_producer_active(True/False)
        # This interaction is the tricky part.
        # Simple strategy:
        # If audio_queue has items -> Active.
        # But FrameGeneratorWorker has an internal buffer too.
        # Let's verify this in a separate manager thread or keep it simple for now.

        # Temporary: Just start them. They might fight for the queue if logic isn't strict.
        # Default: IdlePusher only pushes if queue is low.
        # Producer pushes when it has audio.
        # Conflict: Producer might push 50 frames, but Queue is full of Idle frames.
        # Solution: Interrupt!
        pass

    def push_audio(self, audio_chunk: bytes):
        """
        Entry point for audio stream.
        Triggers "Wake Up" -> "Speaking".
        """
        # 1. Signal "Speaking Mode"
        self.idle_pusher.set_producer_active(True)

        # 2. Add to Producer
        self.audio_input_queue.put(audio_chunk)

        # Logic Hole: When do we set producer_active = False?
        # Ideally, the producer signals "I am empty".
        # We need a callback or a check.

    def end_audio(self):
        """Signal the end of the current PCM speech segment."""
        self.producer.end_audio_stream()

    def interrupt(self):
        """
        Barge-in: Stop everything and return to idle immediately.
        """
        print("IMTalkerEngine: *** INTERRUPT SIGNAL ***")

        # 1. Get the producer's last generated latent BEFORE reset
        # This is more accurate than peek_history() which may have old idle frames
        last_latent = self.producer.last_generated_latent
        last_frame_pixels = self.producer.last_generated_frame
        was_speaking = self.producer.is_speaking

        # 2. Stop processing new audio
        self.producer.hard_reset()

        # 3. Clear pending input audio
        with self.audio_input_queue.mutex:
            self.audio_input_queue.queue.clear()

        # 4. Flush the Frame Queue (Delete all future speaking frames)
        self.frame_queue.purge()

        # 5. Trigger Smart Transition Back to Idle BEFORE enabling idle pusher
        # This prevents race condition where idle frames get pushed before transition
        if last_latent is not None and was_speaking:
            print("IMTalkerEngine: Interrupting from SPEAKING state")
            # NOTE: transition_from_latent will set producer_active=False at the end
            self.idle_pusher.transition_from_latent(last_latent, last_frame_pixels)
        else:
            print("IMTalkerEngine: No speaking latent found or was idle. Just resetting.")
            # Only reset idle pusher if we didn't do a transition
            self.idle_pusher.set_producer_active(False)

        print("IMTalkerEngine: Interrupt Complete.")

    def stop(self):
        print("IMTalkerEngine: Stopping...")
        self.broadcaster.stop()
        self.producer.stop()
        self.idle_pusher.stop()
