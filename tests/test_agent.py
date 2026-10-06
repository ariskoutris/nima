"""Agent decisions, score validation, and short visual memory."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from emulator import Frame
from mystery_dungeon.agent import Agent, DecisionScores, OPTIONS, validate_scores
from mystery_dungeon.session import GameplaySession


FRAME = Frame(4, 4, bytes(4 * 4 * 4))
STILL_FRAME = Frame(16, 16, bytes(16 * 16 * 4))


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


@pytest.mark.parametrize("mode,expected", [("dungeon", None), ("field", "stalled")])
def test_repeated_a_only_stalls_outside_dungeon(tmp_path, mode, expected):
    class Still(FakeEmulator):
        def screenshot(self):
            return STILL_FRAME

    agent = Agent(FakeDecisionModel(), tmp_path, "fake:Model", detect_cycles=True)
    agent.set_goal("Use A")
    session = GameplaySession(Still(), tmp_path)
    session.ram_state = lambda: SimpleNamespace(mode=mode, text_printing=False,
                                                player_position=None)
    session.capture()
    for _ in range(8):
        assert agent.step(session).status == "acted"
    assert agent.pending_loop == expected
    if mode == "dungeon":
        for _ in range(4):
            assert agent.step(session).status == "acted"
    else:
        assert agent.step(session).status == "stalled"


def test_movement_with_changing_ram_position_is_not_a_loop(tmp_path):
    class Walking(FakeEmulator):
        position = 0

        def screenshot(self):
            return STILL_FRAME

        def step(self, frames=1, buttons=()):
            super().step(frames, buttons)
            if buttons:
                self.position += 1

    class Move(FakeDecisionModel):
        index = 0

        def score(self, goal, guidance, frame, history, options):
            button = ("RIGHT", "DOWN")[self.index % 2]
            self.index += 1
            return DecisionScores({key: float(key == button) for key in options},
                                  {"complete": 0, "incomplete": 1, "uncertain": 0})

    game = Walking()
    session = GameplaySession(game, tmp_path)
    session.ram_state = lambda: SimpleNamespace(mode="field", text_printing=False,
                                                player_position=(game.position, 0))
    session.capture()
    agent = Agent(Move(), tmp_path, "fake:Model", detect_cycles=True)
    agent.set_goal("Walk right")
    for _ in range(10):
        assert agent.step(session).status == "acted"
    assert agent.pending_loop is None


def test_repeated_movement_circle_is_detected(tmp_path):
    class Circle(FakeEmulator):
        x = y = 0

        def step(self, frames=1, buttons=()):
            super().step(frames, buttons)
            if buttons:
                direction = next(iter(buttons))
                dx, dy = {"RIGHT": (1, 0), "DOWN": (0, 1),
                          "LEFT": (-1, 0), "UP": (0, -1)}[direction]
                self.x += dx
                self.y += dy

        def screenshot(self):
            return Frame(16, 16, bytes((self.x * 60, self.y * 60, 0, 255)) * 256)

    class Move(FakeDecisionModel):
        index = 0

        def score(self, goal, guidance, frame, history, options):
            button = ("RIGHT", "DOWN", "LEFT", "UP")[self.index % 4]
            self.index += 1
            return DecisionScores({key: float(key == button) for key in options},
                                  {"complete": 0, "incomplete": 1, "uncertain": 0})

    session = GameplaySession(Circle(), tmp_path)
    session.capture()
    agent = Agent(Move(), tmp_path, "fake:Model", detect_cycles=True)
    agent.set_goal("Move around")
    for _ in range(12):
        assert agent.step(session).status == "acted"
    assert agent.pending_loop == "loop"
    assert agent.step(session).status == "loop"
