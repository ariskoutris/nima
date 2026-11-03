"""Serve the live view and coordinate operator input with agent turns."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from emulator import Frame, GbaEmulator
from mystery_dungeon import CORE, KNOWLEDGE, ROM, STATE
from mystery_dungeon.actions import GAME_ACTIONS, GameAction, Stop
from mystery_dungeon.agent import Agent, ModelError, Turn
from mystery_dungeon.model import ChatModel, Model, read_settings
from mystery_dungeon.ram_state import check_rom
from mystery_dungeon.run_log import RunLog
from mystery_dungeon.session import (
    HOLD_TICKS,
    RELEASE_TICKS,
    Checkpoint,
    GameplaySession,
    Observation,
    Transition,
    read_checkpoint,
    read_ss0,
)

COMMANDS = ("start", "pause", "turn", "manual", "reset")
MAX_TASK_LENGTH = 1000
SCREEN_WAIT_SECONDS = 15
TICK_SECONDS = 1 / 60  # one tick at the GBA's refresh rate, to pace streamed frames


@dataclass(frozen=True)
class Manual:
    action: GameAction
    task: str  # the task when the operator acted, so the move stays in that task's memory


@dataclass(frozen=True)
class Load:
    checkpoint: Checkpoint | None  # None reloads the current checkpoint


@dataclass(frozen=True)
class TakeTurn:
    task: str


Request = Manual | Load
Work = Request | TakeTurn


class RealtimeSession:
    """Keep the game running without input from each observation until the next action.

    The game doesn't wait for the model, so an action meets the game as it is when the
    reply arrives, not as the model saw it.
    """

    def __init__(self, session: GameplaySession) -> None:
        self._session = session
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def observe(self) -> Observation:
        observation = self._session.observe()
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return observation

    def apply(self, action: GameAction) -> Transition:
        self.halt()
        return self._session.apply(action)

    def halt(self) -> None:
        """Stop the game after its current tick.

        Raises:
            Exception: the emulator's error if a tick failed while the game ran.
        """
        self._running.clear()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if (error := self._error) is not None:
            self._error = None
            raise error

    def _run(self) -> None:
        try:
            while self._running.is_set():
                self._session.idle()
        except Exception as exc:
            self._error = exc


class LiveRun:
    """Run the emulator on a worker thread and apply operator input between agent turns.

    Input that arrives during a turn waits for it: the turn in progress is applied, then
    requests run in order. Pause, manual actions, reset and load stop the agent after the
    turn in progress.
    """

    def __init__(self, rom: Path, core: Path, state: Path, output: Path,
                 model: Model, hold_ticks: int, release_ticks: int,
                 knowledge: str = "") -> None:
        if output.exists():
            raise FileExistsError(f"session output already exists: {output}")
        self.output = output
        self._rom, self._core = rom, core
        self._checkpoint = read_checkpoint(state)
        self._hold_ticks, self._release_ticks = hold_ticks, release_ticks
        self._log = RunLog(output, model,
                           {"kind": "live", "rom": str(rom), "core": str(core),
                            "initial_state": str(state), "hold_ticks": hold_ticks,
                            "release_ticks": release_ticks}, knowledge)
        self._agent = Agent(self._log.model, knowledge)
        self._condition = threading.Condition()
        self._task = ""
        self._mode = "paused"  # paused | running | one_turn
        self._requests: deque[Request] = deque()
        self._current: Work | None = None
        self._error = self._notice = ""
        self._loaded = self._stopping = False
        self._reset_count = 0
        self._published: tuple[int, int] | None = None  # image index and tick
        self._game_state: dict | None = None
        self._screen_condition = threading.Condition()
        self._screen: Frame | None = None
        self._screen_revision = 0
        self._stream_tick = 0

    def snapshot(self) -> dict:
        with self._condition:
            return {"status": self._status(), "error": self._error, "notice": self._notice,
                    "task": self._task, "running": self._mode != "paused",
                    "reset_count": self._reset_count,
                    "game_state": self._game_state,
                    "history": self._log.recent(),
                    "plan": self._agent.plan, "notes": list(self._agent.notes),
                    "model": self._log.model.name,
                    "actions": list(GAME_ACTIONS), "output": str(self.output)}

    def wait_for_screen(self, after: int | None) -> tuple[bytes, int, int]:
        """Return the screen PNG, its revision and tick once it is newer than ``after``.

        Returns the current screen immediately when ``after`` is None, and after
        SCREEN_WAIT_SECONDS at the latest.
        """
        with self._screen_condition:
            if after is not None:
                self._screen_condition.wait_for(
                    lambda: self._screen_revision > after or self._stopping,
                    timeout=SCREEN_WAIT_SECONDS)
            screen, revision, tick = self._screen, self._screen_revision, self._stream_tick
        return (screen.png_bytes() if screen else b""), revision, tick

    def control(self, data: dict) -> None:
        """Apply operator input from the live view.

        Raises:
            ValueError: when the input is malformed or cannot run yet.
        """
        if not isinstance(data, dict) or set(data) - {"task", "command", "action"}:
            raise ValueError("unknown control field")
        task = data.get("task")
        if task is not None and (not isinstance(task, str) or len(task) > MAX_TASK_LENGTH):
            raise ValueError(f"task must be text of at most {MAX_TASK_LENGTH} characters")
        command = data.get("command")
        if command is not None and command not in COMMANDS:
            raise ValueError("unknown command")
        action = data.get("action")
        if command == "manual" and action not in GAME_ACTIONS:
            raise ValueError("invalid manual action")
        if command != "manual" and action is not None:
            raise ValueError("action requires the manual command")
        with self._condition:
            self._require_loaded()
            task = self._task if task is None else task.strip()
            keeps_running = command is None and self._mode != "paused"
            if not task and (command in ("start", "turn") or keeps_running):
                raise ValueError("enter a task before running the agent")
            self._task = task
            match command:
                case "start" | "turn":
                    self._error = self._notice = ""
                    self._mode = "running" if command == "start" else "one_turn"
                case "pause":
                    self._mode = "paused"
                case "manual":
                    self._queue(Manual(GAME_ACTIONS[action], task))
                case "reset":
                    self._queue(Load(None))
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
            self._queue(Load(checkpoint))
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
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
            session = GameplaySession(emulator, self.output,
                                      self._hold_ticks, self._release_ticks)
            session.load(self._checkpoint)
            self._after_step(session)
            with self._condition:
                self._loaded = True
                self._condition.notify_all()
            while work := self._next():
                try:
                    match work:
                        case Manual(action, task):
                            self._manual(session, action, task)
                        case Load(checkpoint):
                            self._load(session, checkpoint)
                        case TakeTurn(task):
                            self._turn(session, task)
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
                self._current = TakeTurn(self._task)
            return self._current

    def _turn(self, session: GameplaySession, task: str) -> None:
        self._agent.set_task(task)
        realtime = RealtimeSession(session)
        try:
            result, transition = self._agent.act(realtime)
        finally:
            realtime.halt()  # Stop and model errors end the turn without an action
        with self._condition:
            match result:
                case ModelError(message=message):
                    self._log.record_model_error(result)
                    self._mode = "paused"
                    self._error = f"Model failed: {message}"
                # Stop for a task the operator replaced during the turn doesn't end the new one.
                case Turn(action=Stop(outcome)) if result.context.task == self._task:
                    self._log.record_turn(result, transition)
                    self._mode = "paused"
                    self._notice = f"Stopped ({outcome}): {result.reasoning}"
                case Turn():
                    self._log.record_turn(result, transition)
                    if self._mode == "one_turn":
                        self._mode = "paused"
        self._after_step(session)

    def _manual(self, session: GameplaySession, action: GameAction, task: str) -> None:
        self._agent.set_task(task)
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
        self._agent.clear_memory()
        with self._condition:
            self._log.record_reset(session.last_observation.screenshot)
            self._reset_count += 1
            self._error = ""
            self._notice = "Loaded state." if checkpoint is not None else "Reset complete."
        self._after_step(session)

    def _after_step(self, session: GameplaySession) -> None:
        """Save and publish the session's state if the game ran since the last call.

        The live screen shows only streamed ticks; screenshots appear with their action.
        """
        # The tick alone repeats after a reset; the image index alone misses idle ticks.
        if (session.image_index, session.tick) == self._published:
            return
        self._published = (session.image_index, session.tick)
        session.save_state("latest.state")
        with self._condition:
            self._game_state = asdict(session.ram_state())
            self._condition.notify_all()
        with self._screen_condition:
            self._stream_tick = session.tick

    def _fail(self, message: str) -> None:
        with self._condition:
            self._error = message
            self._mode = "paused"
            self._condition.notify_all()

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ValueError(self._error or "emulator is not ready")

    def _queue(self, request: Request) -> None:
        self._mode = "paused"
        self._error = self._notice = ""
        self._requests.append(request)

    def _status(self) -> str:
        if self._error:
            return "error"
        if not self._loaded:
            return "loading"
        match self._current:
            case TakeTurn():
                return "thinking"
            case Load():
                return "resetting"
        return "paused" if self._mode == "paused" else "running"

    def _stream(self, frame: Frame) -> None:
        with self._screen_condition:
            self._stream_tick += 1
            if self._screen is None or frame.rgba != self._screen.rgba:
                self._screen = frame
                self._screen_revision += 1
                self._screen_condition.notify_all()
        time.sleep(TICK_SECONDS)


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
                valid = (request.query.startswith("after=") and after.isascii() and
                         after.isdecimal() and len(after) <= 12)
                if request.query and not valid:
                    self.respond(400, b"Invalid screen revision", "text/plain; charset=utf-8")
                    return
                screen, revision, tick = run.wait_for_screen(int(after) if request.query else None)
                if screen and (not request.query or revision > int(after)):
                    self.respond(200, screen, "image/png", {
                        "X-Screen-Revision": str(revision), "X-Game-Tick": str(tick)})
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
            if request.path not in ("/api/control", "/api/load-state"):
                self.respond(404, b"Not found", "text/plain; charset=utf-8")
                return
            try:
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                length = int(self.headers.get("Content-Length", "0"))
                limit = 16 * 1024 * 1024 if request.path == "/api/load-state" else 16384
                if length < 1 or length > limit:
                    raise ValueError("invalid request size")
                if request.path == "/api/control":
                    if content_type != "application/json":
                        raise ValueError("Content-Type must be application/json")
                    run.control(json.loads(self.rfile.read(length)))
                else:
                    if content_type != "application/octet-stream":
                        raise ValueError("Content-Type must be application/octet-stream")
                    if request.query not in ("format=state", "format=ss0"):
                        raise ValueError("choose a .state or .ss0 file")
                    run.load_state_upload(self.rfile.read(length),
                                          request.query.removeprefix("format="))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self.respond(200, json.dumps(run.snapshot()).encode(), "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, default=ROM)
    parser.add_argument("--core", type=Path, default=CORE)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE,
                        help="text the agent starts out knowing")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hold-ticks", type=int, default=HOLD_TICKS)
    parser.add_argument("--release-ticks", type=int, default=RELEASE_TICKS)
    args = parser.parse_args()
    if min(args.hold_ticks, args.release_ticks) < 1:
        parser.error("tick counts must be positive")
    check_rom(args.rom)
    model = ChatModel(read_settings())  # fails on a missing key before any output exists
    output = args.output or Path("runs/live") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run = LiveRun(args.rom, args.core, args.state, output, model,
                  args.hold_ticks, args.release_ticks,
                  args.knowledge.read_text() if args.knowledge else "")
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
        server.server_close()
        if worker.ident is not None:
            worker.join(timeout=5)


if __name__ == "__main__":
    main()
