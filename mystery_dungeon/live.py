"""Local browser view and controls for a live screenshot model run."""

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
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit

from emulator import BUTTONS, Frame, GbaEmulator
from mystery_dungeon.agent import Agent, load_decision_model
from mystery_dungeon.benchmark import Seed, read_ss0
from mystery_dungeon.session import GameplaySession

ONEJEV = "mystery_dungeon.decision_models.onejev:OneJevDecisionModel"
GLM = "mystery_dungeon.decision_models.glm:GlmDecisionModel"
MODELS = {ONEJEV: "OneJev (local)", GLM: "GLM-5.3-Flash (OpenRouter)"}


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


class LiveRun:
    def __init__(self, rom: Path, core: Path, state: Path, output: Path,
                 model_spec: str, hold_frames: int, release_frames: int, context: str = "",
                 history_frames: int = 1, completion_votes: int = 1,
                 model_server: ModelServer | None = None):
        if output.exists():
            raise FileExistsError(f"session output already exists: {output}")
        self.rom, self.core, self.state, self.output = rom, core, state, output
        self.model_spec = model_spec
        self.model_server = model_server
        self.models = {**MODELS, model_spec: MODELS.get(model_spec, model_spec)}
        self.agent = Agent(load_decision_model(model_spec), output, model_spec,
                           history_frames, completion_votes, context,
                           {"kind": "live", "rom": str(rom), "core": str(core),
                            "initial_state": str(state), "hold_frames": hold_frames,
                            "release_frames": release_frames}, detect_cycles=True)
        self.hold_frames, self.release_frames = hold_frames, release_frames
        self.condition = threading.Condition()
        self.screen_condition = threading.Condition()
        self.pending_goal = ""
        self.pending_steering = ""
        self.pending_history = True
        self.running = False
        self.single_step = False
        self.step_generation = 0
        self.manual: deque[str] = deque()
        self.reset_requested = False
        self.resetting = False
        self.pending_model = model_spec if model_server and model_spec == ONEJEV else None
        self.switching = False
        self.replacement: Seed | bytes | None = None
        self.reset_count = 0
        self.screen_revision = 0
        self.screen = b""
        self.stream_frame = 0
        self.frame = 0
        self.game_state: dict | None = None
        self.loaded = False
        self.busy = False
        self.waiting_for_model = False
        self.error = ""
        self.notice = ""
        self.stopping = False

    @property
    def goal(self) -> str:
        return self.pending_goal

    @property
    def steering(self) -> str:
        return self.pending_steering

    @property
    def use_recent(self) -> bool:
        return self.pending_history

    @property
    def history(self):
        return self.agent.events

    def snapshot(self) -> dict:
        with self.condition:
            status = ("error" if self.error else "loading" if not self.loaded else
                      "waiting" if self.waiting_for_model else
                      "switching" if self.pending_model or self.switching else
                      "resetting" if self.reset_requested or self.resetting else
                      "thinking" if self.busy else "running" if self.running else "paused")
            return {"status": status, "error": self.error, "notice": self.notice,
                    "goal": self.goal, "steering": self.steering,
                    "use_recent": self.use_recent, "frame": self.frame,
                    "screen_revision": self.screen_revision, "reset_count": self.reset_count,
                    "game_state": self.game_state,
                    "history": list(reversed(self.history)),
                    "decision_model": self.model_spec, "requested_model": self.pending_model,
                    "models": self.models, "output": str(self.output)}

    def request_model(self, spec: str) -> None:
        with self.condition:
            if spec not in self.models:
                raise ValueError("unknown decision model")
            if not self.loaded:
                raise ValueError("emulator is not ready")
            if self.pending_model or self.switching or self.reset_requested or self.resetting:
                raise ValueError("another change is in progress")
            if spec == self.model_spec and not self.error:
                return
            self.running = False
            self.single_step = False
            self.manual.clear()
            self.pending_model = spec
            self.error = self.notice = ""
            self.condition.notify_all()

    def control(self, data: dict) -> None:
        if not isinstance(data, dict) or set(data) - {"goal", "steering", "use_recent", "command", "button"}:
            raise ValueError("unknown control field")
        for key in ("goal", "steering"):
            if key in data and (not isinstance(data[key], str) or len(data[key]) > 1000):
                raise ValueError(f"{key} must be text of at most 1000 characters")
        if "use_recent" in data and type(data["use_recent"]) is not bool:
            raise ValueError("use_recent must be a boolean")
        command = data.get("command")
        if command not in (None, "start", "pause", "step", "manual", "reset"):
            raise ValueError("unknown command")
        button = data.get("button")
        if command == "manual" and button not in BUTTONS | {"WAIT"}:
            raise ValueError("invalid manual button")
        if command != "manual" and button is not None:
            raise ValueError("button requires the manual command")
        with self.condition:
            if self.error and not self.loaded:
                raise ValueError(self.error)
            if (self.reset_requested or self.resetting or self.pending_model or self.switching) and command in ("start", "step", "manual", "reset"):
                raise ValueError("a change is still in progress")
            goal = data.get("goal", self.goal).strip()
            if command in ("start", "step") and not goal:
                raise ValueError("set a goal before running")
            if command in ("start", "step", "manual", "reset"):
                self.error = ""
                self.notice = ""
            steering = data.get("steering", self.steering).strip()
            history = data.get("use_recent", self.use_recent)
            changed = (goal, steering, history) != (self.goal, self.steering, self.use_recent)
            self.pending_goal, self.pending_steering, self.pending_history = goal, steering, history
            if changed:
                self.notice = ""
            if command in ("start", "step"):
                self.running = True
                self.single_step = command == "step"
                if command == "step":
                    self.step_generation += 1
            elif command == "pause":
                self.running = False
                self.single_step = False
            elif command == "manual":
                self.running = False
                self.single_step = False
                self.manual.append(button)
            elif command == "reset":
                self.running = False
                self.single_step = False
                self.manual.clear()
                self.reset_requested = True
            self.condition.notify_all()

    def load_state_upload(self, data: bytes, format: str) -> None:
        if format not in ("state", "ss0"):
            raise ValueError("choose a .state or .ss0 file")
        if not data:
            raise ValueError("state file is empty")
        replacement = read_ss0(data) if format == "ss0" else data
        with self.condition:
            if not self.loaded:
                raise ValueError("emulator is not ready")
            if self.reset_requested or self.resetting or self.pending_model or self.switching:
                raise ValueError("another change is in progress")
            self.running = False
            self.single_step = False
            self.manual.clear()
            self.replacement = replacement
            self.reset_requested = True
            self.error = ""
            self.notice = ""
            self.condition.notify_all()

    def _cancel_reason(self) -> str | None:
        if self.stopping:
            return "stopping"
        if self.reset_requested:
            return "reset_requested"
        if self.pending_model or self.switching:
            return "model_change"
        if self.manual:
            return "manual_control"
        if not self.running:
            return "paused"
        return None

    def _model_waiting(self, waiting: bool) -> None:
        with self.condition:
            self.waiting_for_model = waiting
            self.condition.notify_all()

    def _publish_frame(self, frame: Frame, advanced: bool = False) -> None:
        png = frame.png_bytes()
        with self.screen_condition:
            if advanced:
                self.stream_frame += 1
            if png != self.screen:
                self.screen = png
                self.screen_revision += 1
                self.screen_condition.notify_all()

    def _live_frame(self, frame: Frame) -> None:
        self._publish_frame(frame, advanced=True)
        time.sleep(1 / 60)

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            with self.condition:
                self.error = f"{type(exc).__name__}: {exc}"
                self.busy = False
                self.waiting_for_model = False
                self.condition.notify_all()

    def _run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        with GbaEmulator(self.rom, self.core) as emulator:
            emulator.on_frame = self._live_frame
            initial_state = self.state.read_bytes()
            emulator.load_state(initial_state)
            emulator.step()
            session = GameplaySession(emulator, self.output,
                                      self.hold_frames, self.release_frames)
            session.frame_count = 1
            session.capture()
            session.save_latest()
            with self.condition:
                self._publish_frame(session.current)
                self.frame = 1
                state = session.ram_state()
                self.game_state = asdict(state) if state else None
                self.loaded = True
                self.condition.notify_all()

            while True:
                with self.condition:
                    self.busy = False
                    self.condition.wait_for(lambda: self.stopping or self.reset_requested or self.pending_model or self.manual or self.running)
                    if self.stopping:
                        return
                    if self.pending_model:
                        spec = self.pending_model
                        self.switching = True
                        self.busy = False
                        self.condition.notify_all()
                    else:
                        spec = None
                    if spec is None and self.reset_requested:
                        self.reset_requested = False
                        self.resetting = True
                        self.busy = True
                        replacement = self.replacement
                        self.replacement = None
                        previous = session.image_index
                        previous_state = emulator.save_state()
                        (self.output / "states" / f"before-reset-{previous:06d}.state").write_bytes(previous_state)
                        previous_ram = emulator.save_ram() if isinstance(replacement, Seed) else None
                        try:
                            with self.screen_condition:
                                self.stream_frame = 0
                            if isinstance(replacement, Seed):
                                emulator.load_save_ram(replacement.save_ram)
                                emulator.load_state(replacement.state)
                            else:
                                emulator.load_state(replacement if replacement is not None else initial_state)
                            emulator.step()
                        except Exception as exc:
                            if previous_ram is not None:
                                emulator.load_save_ram(previous_ram)
                            emulator.load_state(previous_state)
                            self.error = f"Could not load state: {exc}"
                            self.resetting = False
                            self.busy = False
                            self.condition.notify_all()
                            continue
                        session.frame_count = 1
                        session.capture()
                        session.save_latest()
                        self.agent.reset(session.current_name)
                        session.action_count = 0
                        self.frame = 1
                        state = session.ram_state()
                        self.game_state = asdict(state) if state else None
                        self._publish_frame(session.current)
                        self.running = False
                        self.resetting = False
                        self.reset_count += 1
                        self.busy = False
                        self.error = ""
                        self.notice = "Loaded state. Agent paused." if replacement is not None else "Reset complete. Agent paused."
                        self.condition.notify_all()
                        continue
                    if spec is None:
                        self.agent.set_goal(self.pending_goal)
                        self.agent.set_steering(self.pending_steering)
                        self.agent.set_history_enabled(self.pending_history)
                        manual = self.manual.popleft() if self.manual else None
                        if manual is None:
                            self.agent.resume()
                        active_step = self.step_generation
                        self.busy = True
                if spec is not None:
                    try:
                        if spec == ONEJEV and self.model_server:
                            os.environ["QEV_BASE_URL"] = self.model_server.start()
                        model = load_decision_model(spec)
                    except Exception as exc:
                        if spec == ONEJEV and self.model_server:
                            self.model_server.stop()
                        with self.condition:
                            self.error = f"Could not switch model: {exc}"
                            self.pending_model = None
                            self.switching = False
                            self.condition.notify_all()
                    else:
                        with self.condition:
                            if spec == self.model_spec:
                                self.agent.model = model
                            else:
                                self.agent.change_model(model, spec)
                            self.model_spec = spec
                            self.pending_model = None
                            self.switching = False
                            self.notice = f"Switched to {self.models[spec]}. Agent paused."
                            self.condition.notify_all()
                        if spec != ONEJEV and self.model_server:
                            self.model_server.stop()
                    continue

                if manual is not None:
                    with self.condition:
                        self.agent.manual_step(session, manual)
                        session.save_latest()
                        self.frame = session.frame_count
                        state = session.ram_state()
                        self.game_state = asdict(state) if state else None
                        self._publish_frame(session.current)
                        self.busy = False
                        self.condition.notify_all()
                    continue

                turn = self.agent.step(session, action_lock=self.condition,
                                       cancel_reason=self._cancel_reason,
                                       on_model_waiting=self._model_waiting)
                with self.condition:
                    self.busy = False
                    if session.frame_count != self.frame:
                        session.save_latest()
                        self.frame = session.frame_count
                        state = session.ram_state()
                        self.game_state = asdict(state) if state else None
                        self._publish_frame(session.current)
                    if turn.status == "cancelled":
                        self.condition.notify_all()
                        continue
                    if turn.status in ("acted", "transition"):
                        if (self.single_step and self.step_generation == active_step
                                and not turn.needs_followup):
                            self.running = False
                            self.single_step = False
                    elif turn.status == "completed":
                        if self.pending_goal == self.agent.goal:
                            self.running = False
                            self.single_step = False
                            self.notice = f"Completed: {self.agent.goal}"
                    elif turn.status in ("stalled", "loop"):
                        if self.pending_goal == self.agent.goal:
                            self.running = False
                            self.single_step = False
                            self.notice = (f"Paused: {turn.error}" if turn.error else
                                           f"Paused: {turn.status} without visible progress. Revise the direction.")
                    elif turn.status == "model_error" and self.pending_goal == self.agent.goal:
                        self.error = f"DecisionModel failed: {turn.error}"
                        self.running = False
                        self.single_step = False
                    self.condition.notify_all()


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
                with run.screen_condition:
                    if request.query:
                        run.screen_condition.wait_for(
                            lambda: run.screen_revision > int(after) or run.stopping, timeout=15)
                    screen = run.screen
                    revision = run.screen_revision
                    frame = run.stream_frame or run.frame
                if screen and (not request.query or revision > int(after)):
                    self.respond(200, screen, "image/png", {
                        "X-Screen-Revision": str(revision), "X-Game-Frame": str(frame)})
                elif screen:
                    self.respond(204, b"", "image/png")
                else:
                    self.respond(404, b"Screen is loading", "text/plain; charset=utf-8")
            elif path.startswith("/api/observation/") and observation_number.isascii() \
                    and observation_number.isdecimal() and len(observation_number) <= 12:
                index = int(observation_number)
                screenshot = run.output / "frames" / f"{index:06d}.png"
                if screenshot.is_file():
                    self.respond(200, screenshot.read_bytes(), "image/png")
                else:
                    self.respond(404, b"No such decision", "text/plain; charset=utf-8")
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
                            raise ValueError("model must be a decision model name")
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
    parser.add_argument("--decision-model", default=ONEJEV)
    parser.add_argument("--context", type=Path, help="general controls supplied on every decision")
    parser.add_argument("--url", help="OneJev server URL")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--release-frames", type=int, default=4)
    parser.add_argument("--history-frames", type=int, default=1)
    parser.add_argument("--completion-votes", type=int, default=1)
    args = parser.parse_args()
    if min(args.hold_frames, args.release_frames, args.completion_votes) < 1 or args.history_frames < 0:
        parser.error("frame counts and completion votes must be positive; history frames may be zero")
    if args.url:
        os.environ["QEV_BASE_URL"] = args.url
    output = args.output or Path("runs/live") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    config = tomllib.loads((Path(__file__).parents[1] / "config.toml").read_text())
    model_server = (ModelServer(config["server"]["gguf"], config["server"]["mmproj"], output)
                    if not args.url else None)
    run = LiveRun(args.rom, args.core, args.state, output, args.decision_model,
                  args.hold_frames, args.release_frames, args.context.read_text() if args.context else "",
                  args.history_frames, args.completion_votes, model_server)
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
        with run.condition:
            run.stopping = True
            run.condition.notify_all()
        with run.screen_condition:
            run.screen_condition.notify_all()
        if model_server:
            model_server.close()
        server.server_close()
        if worker.ident is not None:
            worker.join(timeout=5)


if __name__ == "__main__":
    main()
