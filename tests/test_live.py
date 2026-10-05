"""Live controls reject stale decisions and record manual actions."""

import json
import threading

import pytest

from emulator import Frame
from mystery_dungeon.agent import DecisionScores
from mystery_dungeon import live


class FakeEmulator:
    frame = Frame(16, 16, bytes(16 * 16 * 4))

    def __init__(self, *args):
        self.buttons = []
        self.loaded_states = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def load_state(self, data):
        assert data == b"start"
        self.loaded_states.append(data)

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

    def start(model):
        state = tmp_path / "start.state"
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
    assert "repeated" in run.notice
    assert len(model.calls) == 9
    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
    actions = [row for row in rows if row["event"] == "action"]
    assert actions[-1] == run.history[-1]
    assert actions[-1]["stalled"]
    discarded = [row for row in rows if row["event"] == "discarded_decision"]
    assert discarded[-1]["reason"] == "stalled"
    assert discarded[-1]["screenshot"] == actions[-1]["after"]
    assert discarded[-1]["history"][0]["action"] == "DOWN"


def test_changed_goal_discards_inflight_answer(start_run):
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
    assert [call[0] for call in model.calls] == ["Old", "New"]
    assert run.history[0]["goal"] == "New"


def test_changed_goal_ignores_inflight_error(start_run):
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
    discarded = [row for row in rows if row["event"] == "discarded_error"]
    assert discarded[0]["goal"] == "Old"
