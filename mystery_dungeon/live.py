"""Local browser view and controls for a live screenshot model run."""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from emulator import BUTTONS, GbaEmulator
from mystery_dungeon.agent import Agent, load_decision_model
from mystery_dungeon.session import GameplaySession


class LiveRun:
    def __init__(self, rom: Path, core: Path, state: Path, output: Path,
                 model_spec: str, hold_frames: int, release_frames: int, context: str = "",
                 history_frames: int = 1, completion_votes: int = 1):
        if output.exists():
            raise FileExistsError(f"session output already exists: {output}")
        self.rom, self.core, self.state, self.output = rom, core, state, output
        self.model_spec = model_spec
        self.agent = Agent(load_decision_model(model_spec), output, model_spec,
                           history_frames, completion_votes, context,
                           {"kind": "live", "rom": str(rom), "core": str(core),
                            "initial_state": str(state), "hold_frames": hold_frames,
                            "release_frames": release_frames})
        self.hold_frames, self.release_frames = hold_frames, release_frames
        self.condition = threading.Condition()
        self.pending_goal = ""
        self.pending_steering = ""
        self.pending_history = True
        self.running = False
        self.single_step = False
        self.manual: deque[str] = deque()
        self.reset_requested = False
        self.resetting = False
        self.reset_count = 0
        self.screen_revision = 0
        self.screen = b""
        self.frame = 0
        self.loaded = False
        self.busy = False
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
                      "resetting" if self.reset_requested or self.resetting else
                      "thinking" if self.busy else "running" if self.running else "paused")
            return {"status": status, "error": self.error, "notice": self.notice,
                    "goal": self.goal, "steering": self.steering,
                    "use_recent": self.use_recent, "frame": self.frame,
                    "screen_revision": self.screen_revision, "reset_count": self.reset_count,
                    "history": list(reversed(self.history)),
                    "decision_model": self.model_spec, "output": str(self.output)}

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
            if (self.reset_requested or self.resetting) and command in ("start", "step", "manual", "reset"):
                raise ValueError("reset is still in progress")
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

    def _cancel_reason(self) -> str | None:
        if self.stopping:
            return "stopping"
        if self.reset_requested:
            return "reset_requested"
        if self.manual:
            return "manual_control"
        if not self.running:
            return "paused"
        return None

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            with self.condition:
                self.error = f"{type(exc).__name__}: {exc}"
                self.busy = False
                self.condition.notify_all()

    def _run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        with GbaEmulator(self.rom, self.core) as emulator:
            initial_state = self.state.read_bytes()
            emulator.load_state(initial_state)
            emulator.step()
            session = GameplaySession(emulator, self.output,
                                      self.hold_frames, self.release_frames)
            session.frame_count = 1
            session.capture()
            session.save_latest()
            with self.condition:
                self.screen = session.current.png_bytes()
                self.screen_revision = 1
                self.frame = 1
                self.loaded = True
                self.condition.notify_all()

            while True:
                with self.condition:
                    self.busy = False
                    self.condition.wait_for(lambda: self.stopping or self.reset_requested or self.manual or self.running)
                    if self.stopping:
                        return
                    if self.reset_requested:
                        self.reset_requested = False
                        self.resetting = True
                        self.busy = True
                        previous = session.image_index
                        (self.output / "states" / f"before-reset-{previous:06d}.state").write_bytes(
                            emulator.save_state())
                        emulator.load_state(initial_state)
                        emulator.step()
                        session.frame_count = 1
                        session.capture()
                        session.save_latest()
                        self.agent.reset(session.current_name)
                        session.action_count = 0
                        self.frame = 1
                        self.screen = session.current.png_bytes()
                        self.screen_revision += 1
                        self.running = False
                        self.resetting = False
                        self.reset_count += 1
                        self.busy = False
                        self.error = ""
                        self.condition.notify_all()
                        continue
                    self.agent.set_goal(self.pending_goal)
                    self.agent.set_steering(self.pending_steering)
                    self.agent.set_history_enabled(self.pending_history)
                    manual = self.manual.popleft() if self.manual else None
                    if manual is None:
                        self.agent.resume()
                    self.busy = True

                if manual is not None:
                    with self.condition:
                        self.agent.manual_step(session, manual)
                        session.save_latest()
                        self.frame = session.frame_count
                        self.screen = session.current.png_bytes()
                        self.screen_revision += 1
                        self.busy = False
                        self.condition.notify_all()
                    continue

                turn = self.agent.step(session, action_lock=self.condition,
                                       cancel_reason=self._cancel_reason)
                with self.condition:
                    self.busy = False
                    if turn.status == "cancelled":
                        self.condition.notify_all()
                        continue
                    if turn.status in ("acted", "transition"):
                        session.save_latest()
                        self.frame = session.frame_count
                        self.screen = session.current.png_bytes()
                        self.screen_revision += 1
                        if self.single_step and not turn.needs_followup:
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
                            self.notice = f"Paused: {turn.status} without visible progress. Revise the direction."
                    elif turn.status == "model_error" and self.pending_goal == self.agent.goal:
                        self.error = f"DecisionModel failed: {turn.error}"
                        self.running = False
                        self.single_step = False
                    self.condition.notify_all()


def handler_for(run: LiveRun):
    page = (Path(__file__).parent / "static/live.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            observation_number = path.removeprefix("/api/observation/")
            if path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
            elif path == "/api/state":
                self.respond(200, json.dumps(run.snapshot()).encode(), "application/json")
            elif path == "/api/screen":
                with run.condition:
                    screen = run.screen
                if screen:
                    self.respond(200, screen, "image/png")
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
            if urlsplit(self.path).path != "/api/control":
                self.respond(404, b"Not found", "text/plain; charset=utf-8")
                return
            try:
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 16384:
                    raise ValueError("invalid request size")
                data = json.loads(self.rfile.read(length))
                run.control(data)
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
    parser.add_argument("--decision-model", default="mystery_dungeon.decision_models.onejev:OneJevDecisionModel")
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
    run = LiveRun(args.rom, args.core, args.state, output, args.decision_model,
                  args.hold_frames, args.release_frames, args.context.read_text() if args.context else "",
                  args.history_frames, args.completion_votes)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(run))
    threading.Thread(target=run.run, daemon=True).start()
    print(f"Live view: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Session files: {output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with run.condition:
            run.stopping = True
            run.condition.notify_all()
        server.server_close()


if __name__ == "__main__":
    main()
