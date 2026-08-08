from __future__ import annotations

import atexit
import logging
import os
import subprocess
import time

import httpx

from .config import Avtr1Config

logger = logging.getLogger(__name__)


class Avtr1RendererService:
    """Own the isolated Python 3.12 AVTR-1 renderer process."""

    def __init__(self, config: Avtr1Config) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self._healthy():
            logger.info("Using existing AVTR-1 renderer at %s", self.config.renderer_url)
            return
        environment = os.environ.copy()
        environment.setdefault("AVTR1_LOCAL_STORAGE", str(self.config.avtr1_dir / "artifacts"))
        environment["LOAD_BALANCER_URL"] = "disabled"
        port = self.config.renderer_url.rstrip("/").rsplit(":", maxsplit=1)[-1]
        command = [
            str(self.config.pixi),
            "run",
            "-e",
            "renderer",
            "python",
            "-m",
            "uvicorn",
            "avtr1_renderer.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ]
        self._process = subprocess.Popen(command, cwd=self.config.avtr1_dir, env=environment)
        atexit.register(self.stop)
        deadline = time.monotonic() + self.config.renderer_start_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"AVTR-1 renderer exited during startup with code {self._process.returncode}"
                )
            if self._healthy():
                logger.info("AVTR-1 renderer is healthy")
                return
            time.sleep(0.5)
        self.stop()
        raise TimeoutError("AVTR-1 renderer did not become healthy before timeout")

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def _healthy(self) -> bool:
        try:
            response = httpx.get(
                self.config.renderer_url.rstrip("/") + "/health",
                timeout=1.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False
