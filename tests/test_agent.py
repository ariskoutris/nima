"""Agent decisions, score validation, and short visual memory."""

import hashlib
import json
import pytest

from emulator import Frame
from mystery_dungeon.agent import (GLM_MODEL, Agent, DecisionContext, DecisionScores,
                                   OPTIONS, RecentStep, default_context_path,
                                   validate_scores)
from mystery_dungeon.session import GameplaySession


FRAME = Frame(4, 4, bytes(4 * 4 * 4))


class FakeEmulator:
    def __init__(self):
        self.presses = []

    def screenshot(self):
        return FRAME

    def step(self, frames=1, buttons=()):
        self.presses.append((frames, buttons))

    def read_memory(self, address, size):
        raise AssertionError("RAM leaked into the agent")


class FakeDecisionModel:
    def __init__(self, votes=(0.0,)):
        self.votes = iter(votes)
        self.calls = []

    def score(self, goal, guidance, frame, history, options):
        self.calls.append((goal, guidance, frame, history, options))
        vote = next(self.votes, 0.0)
        return DecisionScores({key: float(key == "A") for key in options},
                              {"complete": vote, "incomplete": 1 - vote, "uncertain": 0})


def test_history_and_journal_have_one_owner(tmp_path):
    model = FakeDecisionModel()
    agent = Agent(model, tmp_path, "fake:Model", history_frames=2)
    agent.set_goal("Find exit")
    session = GameplaySession(FakeEmulator(), tmp_path)
    session.capture()
    for button in ("A", "RIGHT"):
        agent.manual_step(session, button)
    turn = agent.step(session)
    assert turn.status == "acted"
    assert [step.action for step in model.calls[-1][3]] == ["A", "RIGHT"]
    assert [step.screenshot for step in turn.decision.history] == [
        "frames/000000.png", "frames/000001.png"]
    assert all((tmp_path / step.screenshot).is_file() for step in turn.decision.history)
    rows = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [row["source"] for row in rows if row["event"] == "action"] == [
        "manual", "manual", "agent"]
    agent.set_goal("Talk")
    assert not agent.history
    assert agent.goal == "Talk"


def test_run_snapshots_prompt_and_guidance(tmp_path):
    model = FakeDecisionModel()
    model.prompt_version = "v2"
    model.prompt_bytes = b'{"version":"v2"}\n'
    Agent(model, tmp_path, "fake:Model", context="  Use the map.  ")
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["format_version"] == 2
    assert config["decision_model"] == "fake:Model"
    assert config["prompt"]["version"] == "v2"
    assert (tmp_path / config["prompt"]["file"]).read_bytes() == model.prompt_bytes
    assert config["prompt"]["sha256"] == hashlib.sha256(model.prompt_bytes).hexdigest()
    assert (tmp_path / config["context"]["file"]).read_text() == "Use the map."


def test_glm_defaults_to_controls_and_explicit_empty_context_disables_it(tmp_path):
    guide = default_context_path(GLM_MODEL).read_text().strip()
    agent = Agent(FakeDecisionModel(), tmp_path / "default", GLM_MODEL)
    assert agent.guidance == guide
    config = json.loads((agent.output / "config.json").read_text())
    assert (agent.output / config["context"]["file"]).read_text() == guide
    agent.change_model(FakeDecisionModel(), "fake:Model")
    assert agent.guidance == ""
    agent.change_model(FakeDecisionModel(), GLM_MODEL)
    assert agent.guidance == guide
    assert Agent(FakeDecisionModel(), tmp_path / "no_guide", GLM_MODEL,
                 context="").guidance == ""


def test_completion_stops_before_input_and_two_votes_continue(tmp_path):
    game = FakeEmulator()
    agent = Agent(FakeDecisionModel((0.6,)), tmp_path, "fake:Model")
    agent.set_goal("Done")
    session = GameplaySession(game, tmp_path)
    session.capture()
    assert agent.step(session).status == "completed"
    assert game.presses == []

    agent = Agent(FakeDecisionModel((0.6, 0.6)), tmp_path, "fake:Model", completion_votes=2)
    agent.set_goal("Done")
    session = GameplaySession(game, tmp_path)
    session.capture()
    assert agent.step(session).status == "acted"
    assert agent.step(session).status == "completed"
    assert len(game.presses) == 2


def test_score_validation():
    buttons = {key: 0.0 for key in OPTIONS}
    _, normalized = validate_scores(DecisionScores(buttons,
        {"complete": 2, "incomplete": 1, "uncertain": 1}))
    assert normalized["complete"] == 0.5
    for completion in ({"complete": -1, "incomplete": 2, "uncertain": 0},
                       {"complete": 0, "incomplete": 0, "uncertain": 0}):
        with pytest.raises(ValueError):
            validate_scores(DecisionScores(buttons, completion))
    with pytest.raises(ValueError):
        validate_scores(DecisionScores({**buttons, "A": float("nan")},
                                       {"complete": 1, "incomplete": 0, "uncertain": 0}))


def test_image_scale_only_changes_model_input(tmp_path):
    left, right = bytes((1, 2, 3, 4)), bytes((5, 6, 7, 8))
    frame = Frame(2, 1, left + right)
    assert frame.scaled_nearest(2) == Frame(4, 2, (left * 2 + right * 2) * 2)
    model = FakeDecisionModel()
    agent = Agent(model, tmp_path, "fake:Model", image_scale=2)
    history = (RecentStep(frame, "LEFT", "previous.png"),)
    decision, _ = agent._decide(frame, DecisionContext("Move", "", history, 0))
    assert model.calls[0][2] == frame.scaled_nearest(2)
    assert model.calls[0][3][0].frame == frame.scaled_nearest(2)
    assert decision.history == history
    assert list(model.calls[0][4]) == ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "SELECT", "WAIT"]
    assert json.loads((tmp_path / "config.json").read_text())["image_scale"] == 2


def test_zero_history_and_half_completion_do_not_stop(tmp_path):
    model = FakeDecisionModel((0.5, 0.5))
    agent = Agent(model, tmp_path, "fake:Model", history_frames=0)
    agent.set_goal("Find exit")
    session = GameplaySession(FakeEmulator(), tmp_path)
    session.capture()
    assert agent.step(session).status == "acted"
    assert agent.step(session).status == "acted"
    assert not model.calls[-1][3]


def test_repeated_actions_do_not_trigger_an_automatic_wait_or_stop(tmp_path):
    class Still(FakeEmulator):
        def screenshot(self):
            return Frame(16, 16, bytes(16 * 16 * 4))

    game = Still()
    model = FakeDecisionModel()
    agent = Agent(model, tmp_path, "fake:Model")
    agent.set_goal("Keep trying")
    session = GameplaySession(game, tmp_path)
    session.capture()
    for _ in range(12):
        assert agent.step(session).status == "acted"
    assert len(model.calls) == 12
    assert game.presses == [(8, ("A",)), (4, ())] * 12
    assert session.warning == "Possible stall: little screen change after 12 A presses"


def test_agent_only_needs_observe_and_button_press(tmp_path):
    session = GameplaySession(FakeEmulator(), tmp_path)
    session.capture()

    class SessionPort:
        @property
        def current_name(self):
            return session.current_name

        def observe(self):
            return session.observe()

        def apply(self, button):
            return session.apply(button)

    agent = Agent(FakeDecisionModel(), tmp_path, "fake:Model")
    agent.set_goal("Try")
    assert agent.step(SessionPort()).status == "acted"


def test_agent_can_choose_wait_without_an_automatic_stall_stop(tmp_path):
    class WaitModel(FakeDecisionModel):
        def score(self, goal, guidance, frame, history, options):
            return DecisionScores({key: float(key == "WAIT") for key in options},
                                  {"complete": 0, "incomplete": 1, "uncertain": 0})

    game = FakeEmulator()
    session = GameplaySession(game, tmp_path)
    session.capture()
    agent = Agent(WaitModel(), tmp_path, "fake:Model")
    agent.set_goal("Watch")
    assert agent.step(session).button == "WAIT"
    assert game.presses == [(8, ()), (4, ())]
    assert session.warning is None
