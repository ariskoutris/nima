"""Serve the live view and coordinate operator input with agent turns."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from emulator import Frame, GbaEmulator
from mystery_dungeon.actions import ACTIONS, Action
from mystery_dungeon.agent import Agent, Turn
from mystery_dungeon.models import load_model
from mystery_dungeon.run_log import RunLog
from mystery_dungeon.session import Checkpoint, GameplaySession, read_checkpoint, read_ss0

ONEJEV = "mystery_dungeon.models.onejev:OneJevModel"
GLM = "mystery_dungeon.models.glm:GlmModel"
MODELS = {ONEJEV: "OneJev (local)", GLM: "GLM-5.3-Flash (OpenRouter)"}
COMMANDS = ("start", "pause", "turn", "manual", "reset")
MAX_INSTRUCTIONS = 1000
SCREEN_WAIT_SECONDS = 15
FRAME_SECONDS = 1 / 60  # pace streamed frames at the GBA's refresh rate
NO_MODEL = "choose a model first"


class ModelServer:
    """Own QEV and its llama-server child for the lifetime of the live view."""

    def __init__(self, gguf: str, mmproj: str, output: Path):
        self.gguf, self.mmproj, self.output = gguf, mmproj, output
        self.proc: subprocess.Popen | None = None
        self.url = ""
        self.closed = False

    def start(self) -> str:
        if self.proc:
            if self.proc.poll() is None:
                return self.url
            self.stop()
        if self.closed:
            raise RuntimeError("live platform is stopping")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "model-server.log").open("ab") as log:
            self.proc = subprocess.Popen([
                sys.executable, "-m", "qev.cli", "serve", "--gguf", self.gguf,
                "--mmproj", self.mmproj, "--host", "127.0.0.1", "--port", str(port),
            ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 300
            proc = self.proc
            while time.monotonic() < deadline:
                if self.closed:
                    raise RuntimeError("live platform is stopping")
                if proc.poll() is not None:
                    raise RuntimeError(f"model server exited; see {self.output / 'model-server.log'}")
                try:
                    with urlopen(self.url + "/v1/models", timeout=1) as response:
                        if response.status == 200:
                            return self.url
                except OSError:
                    pass
                time.sleep(0.5)
            raise TimeoutError("model server did not become ready")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        proc, self.proc = self.proc, None
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


@dataclass(frozen=True)
class Manual:
    action: Action


@dataclass(frozen=True)
class Load:
    checkpoint: Checkpoint | None  # None reloads the current checkpoint


@dataclass(frozen=True)
class SwitchModel:
    spec: str


@dataclass(frozen=True)
class TakeTurn:
    instructions: str


Request = Manual | Load | SwitchModel
Work = Request | TakeTurn


class LiveRun:
    """Run the emulator on a worker thread and apply operator input between agent turns.

    Every operator input except Start and One turn interrupts a turn in progress:
    its result is logged as discarded and nothing is applied. Requests run in order.
    """

    def __init__(self, rom: Path, core: Path, state: Path, output: Path, model_spec: str,
                 hold_frames: int, release_frames: int, knowledge: str = "",
                 model_server: ModelServer | None = None) -> None:
        if output.exists():
            raise FileExistsError(f"session output already exists: {output}")
        self.output = output
        self.models = {**MODELS, model_spec: MODELS.get(model_spec, model_spec)}
        self._rom, self._core = rom, core
        self._checkpoint = read_checkpoint(state)
        self._hold_frames, self._release_frames = hold_frames, release_frames
        self._knowledge = knowledge
        self._model_server = model_server
        self._log = RunLog(output, model_spec,
                           {"kind": "live", "rom": str(rom), "core": str(core),
                            "initial_state": str(state), "hold_frames": hold_frames,
                            "release_frames": release_frames}, knowledge)
        self._agent: Agent | None = None
        self._condition = threading.Condition()
        self._instructions = ""
        self._mode = "paused"  # paused | running | one_turn
        self._requests: deque[Request] = deque([SwitchModel(model_spec)])
        self._current: Work | None = None
        self._interrupted: str | None = None
        self._error = self._notice = ""
        self._loaded = self._stopping = False
        self._reset_count = 0
        self._published_image = -1
        self._game_state: dict | None = None
        self._warning: str | None = None
        self._screen_condition = threading.Condition()
        self._screen: Frame | None = None
        self._screen_revision = 0
        self._stream_frame = 0

    def snapshot(self) -> dict:
        with self._condition:
            return {"status": self._status(), "error": self._error, "notice": self._notice,
                    "instructions": self._instructions, "reset_count": self._reset_count,
                    "game_state": self._game_state, "warning": self._warning,
                    "history": self._log.recent(), "model": self._log.model_spec,
                    "requested_model": self._pending_model(), "models": self.models,
                    "actions": list(ACTIONS), "output": str(self.output)}

    def wait_for_screen(self, after: int | None) -> tuple[bytes, int, int]:
        """Return the screen PNG, its revision and game frame once it is newer than ``after``.

        Returns the current screen immediately when ``after`` is None, and after
        SCREEN_WAIT_SECONDS at the latest.
        """
        with self._screen_condition:
            if after is not None:
                self._screen_condition.wait_for(
                    lambda: self._screen_revision > after or self._stopping,
                    timeout=SCREEN_WAIT_SECONDS)
            screen, revision, frame = self._screen, self._screen_revision, self._stream_frame
        return (screen.png_bytes() if screen else b""), revision, frame

    def control(self, data: dict) -> None:
        """Apply operator input from the live view.

        Raises:
            ValueError: when the input is malformed or cannot run yet.
        """
        if not isinstance(data, dict) or set(data) - {"instructions", "command", "action"}:
            raise ValueError("unknown control field")
        instructions = data.get("instructions")
        if instructions is not None and (not isinstance(instructions, str)
                                         or len(instructions) > MAX_INSTRUCTIONS):
            raise ValueError(f"instructions must be text of at most {MAX_INSTRUCTIONS} characters")
        command = data.get("command")
        if command is not None and command not in COMMANDS:
            raise ValueError("unknown command")
        action = data.get("action")
        if command == "manual" and action not in ACTIONS:
            raise ValueError("invalid manual action")
        if command != "manual" and action is not None:
            raise ValueError("action requires the manual command")
        with self._condition:
            self._require_loaded()
            instructions = self._instructions if instructions is None else instructions.strip()
            keeps_running = command is None and self._mode != "paused"
            if not instructions and (command in ("start", "turn") or keeps_running):
                raise ValueError("enter instructions before running the agent")
            if command in ("start", "turn", "manual") and self._agent is None \
                    and self._pending_model() is None:
                raise ValueError(NO_MODEL)
            if instructions != self._instructions:
                self._instructions = instructions
                self._interrupt("instructions_changed")
            match command:
                case "start" | "turn":
                    self._error = self._notice = ""
                    self._mode = "running" if command == "start" else "one_turn"
                case "pause":
                    self._pause("paused")
                case "manual":
                    self._queue("manual_control", Manual(ACTIONS[action]))
                case "reset":
                    self._queue("reset_requested", Load(None))
            self._condition.notify_all()

    def request_model(self, spec: str) -> None:
        """Switch both model roles to ``spec`` after the current turn is discarded.

        Raises:
            ValueError: for an unknown model or before the emulator is ready.
        """
        with self._condition:
            if spec not in self.models:
                raise ValueError("unknown model")
            self._require_loaded()
            target = self._pending_model() or self._log.model_spec
            if spec == target and self._agent is not None and not self._error:
                return
            self._queue("model_change", SwitchModel(spec))
            self._condition.notify_all()

    def load_state_upload(self, data: bytes, format: str) -> None:
        """Load an uploaded .state or .ss0 checkpoint and make it the reset target.

        Raises:
            ValueError: for an unknown format, unreadable data, or before the emulator is ready.
        """
        if format not in ("state", "ss0"):
            raise ValueError("choose a .state or .ss0 file")
        if not data:
            raise ValueError("state file is empty")
        checkpoint = read_ss0(data) if format == "ss0" else Checkpoint(data)
        with self._condition:
            self._require_loaded()
            self._queue("reset_requested", Load(checkpoint))
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._interrupt("stopping")
            self._condition.notify_all()
        with self._screen_condition:
            self._screen_condition.notify_all()

    def run(self) -> None:
        """Run the worker loop until stop(); failures end the loop and show as an error."""
        try:
            self._run()
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        with GbaEmulator(self._rom, self._core) as emulator:
            emulator.on_frame = self._stream
            session = GameplaySession(emulator, self.output, self._hold_frames, self._release_frames)
            session.load(self._checkpoint)
            self._after_step(session)
            with self._condition:
                self._loaded = True
                self._condition.notify_all()
            while work := self._next():
                try:
                    match work:
                        case Manual(action):
                            self._manual(session, action)
                        case Load(checkpoint):
                            self._load(session, checkpoint)
                        case SwitchModel(spec):
                            self._switch_model(spec)
                        case TakeTurn(instructions):
                            self._turn(session, instructions)
                finally:
                    with self._condition:
                        self._current = None
                        self._condition.notify_all()

    def _next(self) -> Work | None:
        """Wait for the next request, or a turn when the agent should take one; None to stop."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._stopping or self._requests or self._mode != "paused")
            if self._stopping:
                return None
            if self._requests:
                self._current = self._requests.popleft()
            else:
                self._interrupted = None
                self._current = TakeTurn(self._instructions)
            return self._current

    def _turn(self, session: GameplaySession, instructions: str) -> None:
        self._agent.set_instructions(instructions)
        turn = self._agent.propose(session.observe())
        with self._condition:
            if self._interrupted is not None:
                self._log.record_discarded(turn, self._interrupted)
            else:
                self._log.record_turn(turn, self._agent.commit(session, turn))
                self._settle(turn)
        self._after_step(session)

    def _settle(self, turn: Turn) -> None:
        match turn.status:
            case "completion":
                self._mode = "paused"
                self._notice = f"Completed: {turn.task.description}"
            case "model_error":
                self._mode = "paused"
                self._error = f"Model failed: {turn.error}"
            case "action" if self._mode == "one_turn":
                self._mode = "paused"

    def _manual(self, session: GameplaySession, action: Action) -> None:
        if self._agent is None:
            # Keep the failed model switch that left the agent missing visible.
            self._fail(self._error or NO_MODEL)
            return
        transition = self._agent.manual(session, action)
        with self._condition:
            self._log.record_manual(action, transition)
        self._after_step(session)

    def _load(self, session: GameplaySession, checkpoint: Checkpoint | None) -> None:
        session.save_state(f"before-reset-{session.image_index:06d}.state")
        try:
            session.load(checkpoint if checkpoint is not None else self._checkpoint)
        except Exception as exc:
            self._fail(f"Could not load state: {exc}")
            return
        if checkpoint is not None:
            self._checkpoint = checkpoint
        if self._agent is not None:
            self._agent.clear_memory()
        with self._condition:
            self._log.record_reset(session.current_name)
            self._reset_count += 1
            self._error = ""
            self._notice = "Loaded state." if checkpoint is not None else "Reset complete."
        self._after_step(session)

    def _switch_model(self, spec: str) -> None:
        local = spec == ONEJEV and self._model_server is not None
        try:
            if local:
                os.environ["QEV_BASE_URL"] = self._model_server.start()
            model = self._log.watch(load_model(spec))
        except Exception as exc:
            if local:
                self._model_server.stop()
            self._fail(f"Could not switch model: {exc}")
            return
        if not local and self._model_server is not None:
            self._model_server.stop()
        first = self._agent is None
        if first:
            self._agent = Agent(model, model, self._knowledge)
        else:
            self._agent.set_models(model, model)
        with self._condition:
            self._log.record_model_change(spec)
            if not first:
                self._notice = f"Switched to {self.models[spec]}."

    def _after_step(self, session: GameplaySession) -> None:
        """Save and show the session's state if it captured a frame since the last call."""
        if session.image_index == self._published_image:
            return
        self._published_image = session.image_index
        session.save_state("latest.state")
        state = session.ram_state()
        with self._condition:
            self._game_state = asdict(state) if state else None
            self._warning = session.warning
            self._condition.notify_all()
        with self._screen_condition:
            self._stream_frame = session.frame_count
        self._show(session.current)

    def _fail(self, message: str) -> None:
        with self._condition:
            self._error = message
            self._mode = "paused"
            self._condition.notify_all()

    def _pending_model(self) -> str | None:
        return next((work.spec for work in reversed((self._current, *self._requests))
                     if isinstance(work, SwitchModel)), None)

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ValueError(self._error or "emulator is not ready")

    def _pause(self, reason: str) -> None:
        self._mode = "paused"
        self._interrupt(reason)

    def _queue(self, reason: str, request: Request) -> None:
        self._pause(reason)
        self._error = self._notice = ""
        self._requests.append(request)

    def _interrupt(self, reason: str) -> None:
        if self._interrupted is None:
            self._interrupted = reason

    def _status(self) -> str:
        if self._error:
            return "error"
        if not self._loaded:
            return "loading"
        match self._current:
            case TakeTurn():
                return "cancelling" if self._interrupted is not None else "thinking"
            case Load():
                return "resetting"
            case SwitchModel():
                return "switching"
        return "paused" if self._mode == "paused" else "running"

    def _show(self, frame: Frame) -> None:
        with self._screen_condition:
            if self._screen is None or frame.rgba != self._screen.rgba:
                self._screen = frame
                self._screen_revision += 1
                self._screen_condition.notify_all()

    def _stream(self, frame: Frame) -> None:
        with self._screen_condition:
            self._stream_frame += 1
        self._show(frame)
        time.sleep(FRAME_SECONDS)


def handler_for(run: LiveRun):
    page = (Path(__file__).parent / "static/live.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str,
                    headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            path = request.path
            observation_number = path.removeprefix("/api/observation/")
            if path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
            elif path == "/api/state":
                self.respond(200, json.dumps(run.snapshot()).encode(), "application/json")
            elif path == "/api/screen":
                after = request.query.removeprefix("after=")
                if request.query and (not request.query.startswith("after=") or
                                      not after.isascii() or not after.isdecimal() or len(after) > 12):
                    self.respond(400, b"Invalid screen revision", "text/plain; charset=utf-8")
                    return
                screen, revision, frame = run.wait_for_screen(int(after) if request.query else None)
                if screen and (not request.query or revision > int(after)):
                    self.respond(200, screen, "image/png", {
                        "X-Screen-Revision": str(revision), "X-Game-Frame": str(frame)})
                elif screen:
                    self.respond(204, b"", "image/png")
                else:
                    self.respond(404, b"Screen is loading", "text/plain; charset=utf-8")
            elif path.startswith("/api/observation/") and observation_number.isascii() \
                    and observation_number.isdecimal() and len(observation_number) <= 12:
                screenshot = run.output / "frames" / f"{int(observation_number):06d}.png"
                if screenshot.is_file():
                    self.respond(200, screenshot.read_bytes(), "image/png")
                else:
                    self.respond(404, b"No such observation", "text/plain; charset=utf-8")
            else:
                self.respond(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:
            request = urlsplit(self.path)
            if request.path not in ("/api/control", "/api/load-state", "/api/model"):
                self.respond(404, b"Not found", "text/plain; charset=utf-8")
                return
            try:
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > (16 * 1024 * 1024 if request.path == "/api/load-state" else 16384):
                    raise ValueError("invalid request size")
                if request.path in ("/api/control", "/api/model"):
                    if content_type != "application/json":
                        raise ValueError("Content-Type must be application/json")
                    data = json.loads(self.rfile.read(length))
                    if request.path == "/api/model":
                        if not isinstance(data, dict) or set(data) != {"model"} or not isinstance(data["model"], str):
                            raise ValueError("model must be a model name")
                        run.request_model(data["model"])
                    else:
                        run.control(data)
                else:
                    if content_type != "application/octet-stream":
                        raise ValueError("Content-Type must be application/octet-stream")
                    if request.query not in ("format=state", "format=ss0"):
                        raise ValueError("choose a .state or .ss0 file")
                    run.load_state_upload(self.rfile.read(length), request.query.removeprefix("format="))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self.respond(200, json.dumps(run.snapshot()).encode(), "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=ONEJEV)
    parser.add_argument("--knowledge", type=Path, help="text the agent starts out knowing")
    parser.add_argument("--url", help="OneJev server URL")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--release-frames", type=int, default=4)
    args = parser.parse_args()
    if min(args.hold_frames, args.release_frames) < 1:
        parser.error("frame counts must be positive")
    if args.url:
        os.environ["QEV_BASE_URL"] = args.url
    output = args.output or Path("runs/live") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    config = tomllib.loads((Path(__file__).parents[1] / "config.toml").read_text())
    model_server = (ModelServer(config["server"]["gguf"], config["server"]["mmproj"], output)
                    if not args.url else None)
    run = LiveRun(args.rom, args.core, args.state, output, args.model,
                  args.hold_frames, args.release_frames,
                  args.knowledge.read_text() if args.knowledge else "", model_server)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(run))
    worker = threading.Thread(target=run.run, daemon=True)
    try:
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        worker.start()
        print(f"Live view: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Session files: {output}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        run.stop()
        if model_server:
            model_server.close()
        server.server_close()
        if worker.ident is not None:
            worker.join(timeout=5)


if __name__ == "__main__":
    main()
