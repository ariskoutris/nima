"""Run a configured model's serve command and hand out its URL."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from mystery_dungeon.models import CONFIG, ModelEntry

READY_SECONDS = 300  # a first run may download the model before it listens


def serve_command(entry: ModelEntry, port: int) -> list[str]:
    """The entry's serve command with {port} and {python} filled in; other braces stay literal."""
    return [arg.replace("{port}", str(port)).replace("{python}", sys.executable)
            for arg in entry.serve]


class ModelServer:
    """Own at most one model server process at a time, logging to model-server.log."""

    def __init__(self, output: Path):
        self.output = output
        self.proc: subprocess.Popen | None = None
        self.entry: ModelEntry | None = None
        self.url = ""
        self.closed = False

    def start(self, entry: ModelEntry) -> str | None:
        """Return the URL to pass to the entry's adapter.

        An entry with a serve command gets a ready server of its own, replacing any
        other. Any other entry gets its configured url and leaves the running server
        alone, so the caller stops it once the new model has loaded.

        Raises:
            RuntimeError: when the server exits or the owner is closing.
            TimeoutError: when the server does not answer its ready path in time.
        """
        if not entry.serve:
            return entry.url
        if self.proc:
            if self.entry == entry and self.proc.poll() is None:
                return self.url
            self.stop()
        if self.closed:
            raise RuntimeError("model server is closing")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url, self.entry = f"http://127.0.0.1:{port}", entry
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "model-server.log").open("ab") as log:
            # Relative paths in serve commands are relative to config.toml.
            self.proc = subprocess.Popen(serve_command(entry, port), cwd=CONFIG.parent, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + READY_SECONDS
            proc = self.proc
            while time.monotonic() < deadline:
                if self.closed:
                    raise RuntimeError("model server is closing")
                if proc.poll() is not None:
                    raise RuntimeError(f"model server exited; see {self.output / 'model-server.log'}")
                try:
                    with urlopen(self.url + entry.ready, timeout=1):  # raises until ready
                        return self.url
                except OSError:
                    pass
                time.sleep(0.5)
            raise TimeoutError("model server did not become ready")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        proc, self.proc, self.entry, self.url = self.proc, None, None, ""
        if proc:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

    def close(self) -> None:
        self.closed = True
        self.stop()

