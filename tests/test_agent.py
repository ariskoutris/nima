"""Agent decisions, score validation, and short visual memory."""

import hashlib
import json

import pytest

from emulator import Frame
from mystery_dungeon.agent import Agent, DecisionScores, OPTIONS, validate_scores
from mystery_dungeon.session import GameplaySession


FRAME = Frame(4, 4, bytes(4 * 4 * 4))


class FakeEmulator:
    def __init__(self):
        self.presses = []

    def screenshot(self):
        return FRAME

    def step(self, frames=1, buttons=()):
        self.presses.append((frames, buttons))

    def system_ram(self):
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


def test_zero_history_and_half_completion_do_not_stop(tmp_path):
    model = FakeDecisionModel((0.5, 0.5))
    agent = Agent(model, tmp_path, "fake:Model", history_frames=0)
    agent.set_goal("Find exit")
    session = GameplaySession(FakeEmulator(), tmp_path)
    session.capture()
    assert agent.step(session).status == "acted"
    assert agent.step(session).status == "acted"
    assert not model.calls[-1][3]
    agent.transition_wait(session.current_name)
    assert agent.streak == 0
