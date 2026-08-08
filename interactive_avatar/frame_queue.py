import queue
import threading

from .events import OutputFrame


class FrameQueue:
    """
    A thread-safe queue for OutputFrames that supports:
    1. Standard put/get for streaming.
    2. Peeking at history (most recently consumed frames) for transitions.
    3. Purging (clearing future frames) for interruptions.
    """

    def __init__(self, max_size: int = 100, history_size: int = 10):
        self._queue = queue.Queue(maxsize=max_size)
        self._history: list[OutputFrame] = []
        self._history_size = history_size
        self._lock = threading.Lock()  # For protecting history access/updates separate from queue

    def put(self, frame: OutputFrame):
        """Put a frame into the queue. Blocking if full."""
        self._queue.put(frame)

    def get(self, block: bool = True, timeout: float | None = None) -> OutputFrame:
        """
        Get a frame fram the queue.
        Updates the internal history buffer with this consumed frame.
        """
        frame = self._queue.get(block=block, timeout=timeout)

        self.remember(frame)

        return frame

    def remember(self, frame: OutputFrame):
        """Record a frame published outside the queue, such as idle fallback."""
        with self._lock:
            self._history.append(frame)
            if len(self._history) > self._history_size:
                self._history.pop(0)

    def get_nowait(self) -> OutputFrame:
        """Get a frame without blocking."""
        return self.get(block=False)

    def peek_history(self) -> OutputFrame | None:
        """
        Return the most recently consumed frame (what is likely on screen right now).
        Used by the Generator to seamlessly transition.
        """
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]

    def purge(self):
        """
        Clear all pending frames in the queue.
        Used when an interruption occurs.
        """
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.unfinished_tasks = 0
            self._queue.not_full.notify_all()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
