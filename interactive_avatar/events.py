from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch


class FrameType(Enum):
    IDLE = auto()
    SPEAKING = auto()
    TRANSITION = auto()


@dataclass
class OutputFrame:
    """
    Container for a single frame of output.
    Carries both the display data (pixels/audio) and the conditioning data (latents).
    """

    # 1. Display Data (Cheap/Free)
    video_frame: np.ndarray  # [H, W, 3] RGB or BGR, uint8. Ready to show.
    audio_frame: np.ndarray | None  # Audio chunk (PCM). None if silent.

    # 2. Conditioning Data (Invisible Metadata)
    # The 'mesh' encoding or motion latent used to generate this frame.
    # Used by the Generator to condition the Next frame (Transition).
    motion_latent: torch.Tensor  # [1, D]

    # 3. Metadata
    type: FrameType
    timestamp: float = 0.0  # Presentation timestamp
    final_chunk: bool = False  # True if this is the last frame of a speech segment
