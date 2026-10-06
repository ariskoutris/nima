"""Live controls queue goal changes and record manual actions."""

import json
import threading
import tomllib
from contextlib import contextmanager
from pathlib import Path

import pytest

from emulator import Frame, GbaEmulator
from mystery_dungeon.agent import DecisionScores
from mystery_dungeon import live


class FakeEmulator:
    frame = Frame(16, 16, bytes(16 * 16 * 4))
    instances = []

    def __init__(self, *args):
        self.buttons = []
        self.loaded_states = []
        self.ram = b"original-ram"
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def load_state(self, data):
        if data == b"bad":
            raise RuntimeError("invalid core state")
        assert data in (b"start", b"uploaded", b"ss0-state", b"checkpoint")
        self.loaded_states.append(data)

    def save_ram(self):
        return self.ram

    def load_save_ram(self, data):
        self.ram = data

    def step(self, frames=1, buttons=()):
        self.buttons.append(tuple(buttons))

    def screenshot(self):
        return self.frame

    def save_state(self):
        return b"checkpoint"


class FakeDecisionModel:
    def __init__(self, button="A", complete=0.0):
        self.button = button
        self.complete = complete
        self.calls = []

    def score(self, goal, guidance, frame, history, options):
        self.calls.append((goal, guidance, history))
        return DecisionScores({key: float(key == self.button) for key in options},
                              {"complete": self.complete,
                               "incomplete": 1 - self.complete, "uncertain": 0})


@pytest.fixture
def start_run(tmp_path, monkeypatch):
    active = []

    def start(model, state_format="state"):
        state = tmp_path / f"start.{state_format}"
        state.write_bytes(b"start")
        monkeypatch.setattr(live, "load_decision_model", lambda spec: model)
        monkeypatch.setattr(live, "GbaEmulator", FakeEmulator)
        run = live.LiveRun(tmp_path / "game.gba", tmp_path / "core", state,
                           tmp_path / "output", "fake:Model", 8, 4)
        live.handler_for(run)  # The static page must resolve from its new location.
        thread = threading.Thread(target=run.run)
        active.append((run, thread))
        thread.start()
        with run.condition:
            assert run.condition.wait_for(lambda: run.loaded or run.error, 2), run.error
        return run

    yield start
    for run, thread in active:
        with run.condition:
            run.stopping = True
            run.condition.notify_all()
        thread.join(2)
        assert not thread.is_alive()


def test_default_checkpoint_is_square():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "config.toml").read_text())
    assert config["game"]["state"] == "data/game/checkpoints/square.ss0"


def test_ss0_startup_and_reset_restore_save_ram(start_run, monkeypatch):
    monkeypatch.setattr(live, "read_ss0", lambda path: live.Seed(b"ss0-state", b"new-ram"))
    run = start_run(FakeDecisionModel(), "ss0")
    emulator = FakeEmulator.instances[-1]
    assert emulator.loaded_states == [b"ss0-state"]
    assert emulator.ram == b"new-ram"
    emulator.ram = b"changed-ram"
    run.control({"command": "reset"})
    with run.condition:
        assert run.condition.wait_for(lambda: run.reset_count == 1, 2)
    assert emulator.loaded_states[-1] == b"ss0-state"
    assert emulator.ram == b"new-ram"


def test_manual_action_enters_history_and_reset_keeps_goal(start_run):
    model = FakeDecisionModel()
    run = start_run(model)
    run.control({"goal": "Find exit", "command": "manual", "button": "RIGHT"})
    with run.condition:
        assert run.condition.wait_for(lambda: len(run.history) == 1, 2)
    run.control({"command": "step"})
    with run.condition:
        assert run.condition.wait_for(lambda: len(run.history) == 2, 2)
    assert model.calls[0][2][0].action == "RIGHT"
    assert run.history[0]["source"] == "manual"
    assert (run.output / run.history[0]["before"]).is_file()
    assert (run.output / "states/latest.state").read_bytes() == b"checkpoint"
    config = json.loads((run.output / "config.json").read_text())
    assert config["run"]["kind"] == "live"
    assert config["run"]["hold_frames"] == 8
    run.control({"command": "reset"})
    with run.condition:
        assert run.condition.wait_for(lambda: run.reset_count == 1, 2)
    assert run.goal == "Find exit"
    assert not run.agent.history
    assert not run.history
    run.control({"command": "step"})
    with run.condition:
        assert run.condition.wait_for(lambda: len(run.history) == 1, 2)
    assert run.history[0]["step"] == 1
    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
    assert [row for row in rows if row["event"] == "action"][-1]["step"] == 1


def test_upload_state_and_ss0_from_live_view(start_run, monkeypatch):
    run = start_run(FakeDecisionModel())
    run.control({"goal": "Find exit", "command": "manual", "button": "RIGHT"})
    with run.condition:
        assert run.condition.wait_for(lambda: len(run.history) == 1, 2)
    run.load_state_upload(b"uploaded", "state")
    with run.condition:
        assert run.condition.wait_for(lambda: run.reset_count == 1, 2)
    assert FakeEmulator.instances[-1].loaded_states[-1] == b"uploaded"
    assert run.goal == "Find exit"
    assert not run.history
    assert run.notice == "Loaded state. Agent paused."
    run.control({"command": "reset"})
    with run.condition:
        assert run.condition.wait_for(lambda: run.reset_count == 2, 2)
    assert FakeEmulator.instances[-1].loaded_states[-1] == b"uploaded"

    monkeypatch.setattr(live, "read_ss0", lambda data: live.Seed(b"ss0-state", b"new-ram"))
    run.load_state_upload(b"png-data", "ss0")
    with run.condition:
        assert run.condition.wait_for(lambda: run.reset_count == 3, 2)
    assert FakeEmulator.instances[-1].loaded_states[-1] == b"ss0-state"
    assert FakeEmulator.instances[-1].ram == b"new-ram"
    FakeEmulator.instances[-1].ram = b"changed-ram"
    run.control({"command": "reset"})
    with run.condition:
        assert run.condition.wait_for(lambda: run.reset_count == 4, 2)
    assert FakeEmulator.instances[-1].loaded_states[-1] == b"ss0-state"
    assert FakeEmulator.instances[-1].ram == b"new-ram"

    run.load_state_upload(b"bad", "state")
    with run.condition:
        assert run.condition.wait_for(lambda: bool(run.error), 2)
    assert run.reset_count == 4
    assert FakeEmulator.instances[-1].loaded_states[-1] == b"checkpoint"
    run.control({"command": "reset"})
    with run.condition:
        assert run.condition.wait_for(lambda: run.reset_count == 5, 2)
    assert FakeEmulator.instances[-1].loaded_states[-1] == b"ss0-state"
    with pytest.raises(ValueError, match="choose a"):
        run.load_state_upload(b"uploaded", "unknown")


def test_switch_model_cancels_decision_and_stops_local_server(start_run, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class Blocking(FakeDecisionModel):
        def score(self, goal, guidance, frame, history, options):
            entered.set()
            assert release.wait(2)
            return super().score(goal, guidance, frame, history, options)

    class Server:
        stopped = False

        def start(self):
            return "http://127.0.0.1:9999"

        def stop(self):
            self.stopped = True

    run = start_run(Blocking())
    server = Server()
    run.model_server = server
    replacement = FakeDecisionModel(button="B")
    monkeypatch.setattr(live, "load_decision_model", lambda spec: replacement)
    run.control({"goal": "Move", "command": "step"})
    assert entered.wait(2)
    run.request_model(live.ONEJEV)
    release.set()
    with run.condition:
        assert run.condition.wait_for(lambda: run.model_spec == live.ONEJEV and not run.switching, 2)
    assert not run.history
    assert run.agent.model is replacement
    assert run.agent.model_spec == live.ONEJEV
    assert not run.running
    run.request_model(live.GLM)
    with run.condition:
        assert run.condition.wait_for(lambda: run.model_spec == live.GLM and not run.switching, 2)
    assert server.stopped
    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
    assert [row["to"] for row in rows if row["event"] == "model_change"] == [live.ONEJEV, live.GLM]


def test_model_server_owns_process_group(tmp_path, monkeypatch):
    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 9000)

    class Process:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout == 10

    @contextmanager
    def ready(url, timeout):
        assert url == "http://127.0.0.1:9000/v1/models"
        yield type("Response", (), {"status": 200})()

    calls = []
    monkeypatch.setattr(live.socket, "socket", Socket)
    monkeypatch.setattr(live.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs)) or Process())
    monkeypatch.setattr(live, "urlopen", ready)
    monkeypatch.setattr(live.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    server = live.ModelServer("model.gguf", "vision.gguf", tmp_path)
    assert server.start() == "http://127.0.0.1:9000"
    assert calls[0][0][1:4] == ["-m", "qev.cli", "serve"]
    assert calls[0][1]["start_new_session"] is True
    server.close()
    assert calls[-1] == (1234, live.signal.SIGTERM)


def test_live_frames_publish_during_emulator_steps(start_run):
    run = start_run(FakeDecisionModel())
    game = object.__new__(GbaEmulator)

    class Session:
        count = 0

        def run(self):
            self.count += 1

    game._session = Session()
    game._buttons = set()
    game.on_frame = lambda frame: run._publish_frame(frame, advanced=True)
    game.screenshot = lambda: Frame(1, 1, bytes((game._session.count, 0, 0, 255)))
    with run.condition:
        thread = threading.Thread(target=lambda: game.step(3))
        thread.start()
        thread.join(1)
        assert not thread.is_alive(), "live frames must not wait for the action lock"
    assert game._session.count == 3
    assert run.stream_frame == 3
    assert run.screen_revision == 4
    assert run.screen == game.screenshot().png_bytes()
    handler = object.__new__(live.handler_for(run))
    handler.path = "/api/screen?after=3"
    responses = []
    handler.respond = lambda *args: responses.append(args)
    handler.do_GET()
    assert responses[0][0:3] == (200, run.screen, "image/png")
    assert responses[0][3] == {"X-Screen-Revision": "4", "X-Game-Frame": "3"}


def test_completion_stops_before_press(start_run):
    run = start_run(FakeDecisionModel(complete=0.9))
    run.control({"goal": "Done", "command": "step"})
    with run.condition:
        assert run.condition.wait_for(lambda: run.notice.startswith("Completed"), 2)
    assert not run.history
    assert run.agent.completed


def test_stalled_eighth_action_gets_completion_check(start_run):
    model = FakeDecisionModel(button="DOWN")
    run = start_run(model)
    run.control({"goal": "Move down", "command": "start"})
    with run.condition:
        assert run.condition.wait_for(lambda: bool(run.notice), 2)
    assert len(run.history) == 8
    assert "stalled" in run.notice
    assert len(model.calls) == 9
    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
    actions = [row for row in rows if row["event"] == "action"]
    assert actions[-1] == run.history[-1]
    assert actions[-1]["stalled"]
    discarded = [row for row in rows if row["event"] == "discarded_decision"]
    assert discarded[-1]["reason"] == "stalled"
    assert discarded[-1]["screenshot"] == actions[-1]["after"]
    assert discarded[-1]["history"][0]["action"] == "DOWN"


def test_changed_goal_takes_effect_after_current_action(start_run):
    entered, release = threading.Event(), threading.Event()

    class Blocking(FakeDecisionModel):
        def score(self, goal, guidance, frame, history, options):
            if not self.calls:
                entered.set()
                assert release.wait(2)
            return super().score(goal, guidance, frame, history, options)

    model = Blocking()
    run = start_run(model)
    run.control({"goal": "Old", "command": "step"})
    assert entered.wait(2)
    run.control({"goal": "New"})
    release.set()
    with run.condition:
        assert run.condition.wait_for(lambda: len(run.history) == 1, 2)
        assert run.history[0]["goal"] == "Old"
        run.control({"command": "step"})
    with run.condition:
        assert run.condition.wait_for(lambda: len(run.history) == 2, 2)
    assert [call[0] for call in model.calls] == ["Old", "New"]
    assert run.history[-1]["goal"] == "New"
    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
    assert not any(row["event"] == "discarded_decision" for row in rows)


def test_running_agent_applies_queued_goal_on_next_decision(start_run):
    entered, release = threading.Event(), threading.Event()

    class Blocking(FakeDecisionModel):
        def score(self, goal, guidance, frame, history, options):
            if goal == "Old":
                entered.set()
                assert release.wait(2)
            self.complete = float(goal == "New")
            return super().score(goal, guidance, frame, history, options)

    model = Blocking()
    run = start_run(model)
    run.control({"goal": "Old", "command": "start"})
    assert entered.wait(2)
    run.control({"goal": "New"})
    release.set()
    with run.condition:
        assert run.condition.wait_for(lambda: run.notice == "Completed: New", 2)
    assert [call[0] for call in model.calls] == ["Old", "New"]
    assert [row["goal"] for row in run.history] == ["Old"]


def test_pause_during_inference_prevents_press(start_run):
    entered, release = threading.Event(), threading.Event()

    class Blocking(FakeDecisionModel):
        def score(self, goal, guidance, frame, history, options):
            entered.set()
            assert release.wait(2)
            return super().score(goal, guidance, frame, history, options)

    run = start_run(Blocking())
    run.control({"goal": "Move", "command": "start"})
    assert entered.wait(2)
    assert run.snapshot()["status"] == "waiting"
    run.control({"command": "pause"})
    release.set()
    with run.condition:
        assert run.condition.wait_for(lambda: not run.busy, 2)
    assert run.snapshot()["status"] == "paused"
    assert not run.history
    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "discarded_decision"
    assert rows[-1]["reason"] == "paused"


def test_changed_goal_retries_after_current_model_error(start_run):
    entered, release = threading.Event(), threading.Event()

    class FailingOldGoal(FakeDecisionModel):
        def score(self, goal, guidance, frame, history, options):
            if goal == "Old":
                entered.set()
                assert release.wait(2)
                raise RuntimeError("old goal failed")
            return super().score(goal, guidance, frame, history, options)

    run = start_run(FailingOldGoal())
    run.control({"goal": "Old", "command": "step"})
    assert entered.wait(2)
    run.control({"goal": "New"})
    release.set()
    with run.condition:
        assert run.condition.wait_for(lambda: len(run.history) == 1, 2)
    assert run.history[0]["goal"] == "New"
    assert not run.error
    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
    errors = [row for row in rows if row["event"] == "model_error"]
    assert errors[0]["goal"] == "Old"
