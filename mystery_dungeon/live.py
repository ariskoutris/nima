"""Local browser view and controls for a live screenshot classifier run."""

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
from mystery_dungeon.play import choose, load_classifier


class LiveRun:
    def __init__(self, rom: Path, core: Path, state: Path, output: Path,
                 classifier_spec: str, hold_frames: int, release_frames: int):
        self.rom, self.core, self.state, self.output = rom, core, state, output
        self.classifier_spec = classifier_spec
        self.classifier = load_classifier(classifier_spec)
        self.hold_frames, self.release_frames = hold_frames, release_frames
        self.condition = threading.Condition()
        self.goal = ""
        self.steering = ""
        self.running = False
        self.single_step = False
        self.manual: deque[str] = deque()
        self.reset_requested = False
        self.resetting = False
        self.reset_count = 0
        self.revision = 0
        self.screen_revision = 0
        self.screen = b""
        self.frame = 0
        self.history: deque[dict] = deque(maxlen=40)
        self.recent: list[str] = []
        self.loaded = False
        self.busy = False
        self.error = ""
        self.stopping = False

    def snapshot(self) -> dict:
        with self.condition:
            status = ("error" if self.error else "loading" if not self.loaded else
                      "resetting" if self.reset_requested or self.resetting else
                      "thinking" if self.busy else "running" if self.running else "paused")
            return {"status": status, "error": self.error, "goal": self.goal,
                    "steering": self.steering, "frame": self.frame,
                    "screen_revision": self.screen_revision, "reset_count": self.reset_count,
                    "history": list(reversed(self.history)),
                    "classifier": self.classifier_spec,
                    "output": str(self.output)}

    def control(self, data: dict) -> None:
        if not isinstance(data, dict) or set(data) - {"goal", "steering", "command", "button"}:
            raise ValueError("unknown control field")
        for key in ("goal", "steering"):
            if key in data and (not isinstance(data[key], str) or len(data[key]) > 1000):
                raise ValueError(f"{key} must be text of at most 1000 characters")
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
            self.goal = goal
            self.steering = data.get("steering", self.steering).strip()
            if command == "start":
                self.running = True
                self.single_step = False
            elif command == "step":
                self.running = True
                self.single_step = True
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
            self.revision += 1
            self.condition.notify_all()

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            with self.condition:
                self.error = f"{type(exc).__name__}: {exc}"
                self.busy = False
                self.condition.notify_all()

    def _run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=False)
        with GbaEmulator(self.rom, self.core) as emulator, (self.output / "decisions.jsonl").open("w") as log:
            initial_state = self.state.read_bytes()
            emulator.load_state(initial_state)
            emulator.step()
            initial_screen = emulator.screenshot().png_bytes()
            (self.output / "000000.png").write_bytes(initial_screen)
            (self.output / "latest.state").write_bytes(emulator.save_state())
            with self.condition:
                self.screen = initial_screen
                self.screen_revision = 1
                self.frame = 1
                self.loaded = True
                self.condition.notify_all()

            index = 0
            step_number = 0
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
                        reset = True
                    else:
                        reset = False
                        manual = self.manual.popleft() if self.manual else None
                        goal, steering = self.goal, self.steering
                        revision = self.revision
                        frame_before = self.frame
                        screenshot = f"{index:06d}.png"
                        self.busy = True
                if reset:
                    (self.output / f"before-reset-{index:06d}.state").write_bytes(emulator.save_state())
                    emulator.load_state(initial_state)
                    emulator.step()
                    index += 1
                    screen = emulator.screenshot().png_bytes()
                    filename = f"{index:06d}.png"
                    (self.output / filename).write_bytes(screen)
                    checkpoint = self.output / "latest.state.tmp"
                    checkpoint.write_bytes(emulator.save_state())
                    checkpoint.replace(self.output / "latest.state")
                    log.write(json.dumps({"event": "reset", "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                                          "screenshot": filename, "saved_state": f"before-reset-{index - 1:06d}.state"}) + "\n")
                    log.flush()
                    with self.condition:
                        self.frame = 1
                        self.screen = screen
                        self.screen_revision += 1
                        self.history.clear()
                        self.recent.clear()
                        self.running = False
                        self.resetting = False
                        self.reset_count += 1
                        self.busy = False
                        self.error = ""
                        self.condition.notify_all()
                    step_number = 0
                    continue
                if manual is None:
                    prompt = goal + ("\nOperator guidance: " + steering if steering else "")
                    try:
                        decision = choose(self.classifier, self.output / screenshot, prompt, self.recent)
                    except Exception as exc:
                        with self.condition:
                            self.error = f"Classifier failed: {type(exc).__name__}: {exc}"
                            self.running = False
                            self.single_step = False
                            self.busy = False
                            self.condition.notify_all()
                        continue
                    with self.condition:
                        if self.revision != revision or not self.running or self.stopping:
                            self.busy = False
                            continue
                    source = "agent"
                else:
                    decision = {"choice": manual, "scores": {}, "latency_ms": 0}
                    source = "manual"

                button = decision["choice"]
                emulator.step(self.hold_frames, () if button == "WAIT" else (button,))
                emulator.step(self.release_frames)
                after = emulator.screenshot().png_bytes()
                next_screenshot = f"{index + 1:06d}.png"
                (self.output / next_screenshot).write_bytes(after)
                checkpoint = self.output / "latest.state.tmp"
                checkpoint.write_bytes(emulator.save_state())
                checkpoint.replace(self.output / "latest.state")
                row = {"index": index, "step": step_number + 1,
                       "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                       "frame_before": frame_before,
                       "frame_after": frame_before + self.hold_frames + self.release_frames,
                       "screenshot": screenshot, "goal": goal, "steering": steering,
                       "source": source, **decision}
                log.write(json.dumps(row) + "\n")
                log.flush()
                with self.condition:
                    self.frame = row["frame_after"]
                    self.screen = after
                    self.screen_revision += 1
                    self.history.append(row)
                    self.recent.append(button)
                    self.recent = self.recent[-4:]
                    self.busy = False
                    if self.single_step:
                        self.running = False
                        self.single_step = False
                    self.condition.notify_all()
                index += 1
                step_number += 1


def handler_for(run: LiveRun):
    page = Path(__file__).with_name("live.html").read_bytes()

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
                screenshot = run.output / f"{index:06d}.png"
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
    parser.add_argument("--classifier", default="mystery_dungeon.classifiers.onejev:classify")
    parser.add_argument("--url", help="OneJev server URL")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--release-frames", type=int, default=4)
    args = parser.parse_args()
    if min(args.hold_frames, args.release_frames) < 1:
        parser.error("frame counts must be positive")
    if args.url:
        os.environ["QEV_BASE_URL"] = args.url
    output = args.output or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run = LiveRun(args.rom, args.core, args.state, output, args.classifier,
                  args.hold_frames, args.release_frames)
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
