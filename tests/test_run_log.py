"""Run configuration, turn rows and model call rows."""

import hashlib
import json

import pytest
from fakes import ScriptedModel, finding, rows, start, votes

from mystery_dungeon.actions import Press
from mystery_dungeon.agent import Agent
from mystery_dungeon.model import ScratchpadEdit
from mystery_dungeon.run_log import FORMAT_VERSION, RECENT_ROWS, RunLog


def turns(tmp_path):
    return rows(tmp_path / "run/turns.jsonl")


def model_calls(tmp_path):
    return rows(tmp_path / "run/model_calls.jsonl")


def setup(tmp_path, model, knowledge=""):
    log = RunLog(tmp_path / "run", model, {"kind": "test"}, knowledge)
    agent = Agent(log.model, knowledge)
    agent.set_task("Find exit")
    return log, agent, start(tmp_path / "run")


def test_config_snapshots_knowledge(tmp_path):
    RunLog(tmp_path, ScriptedModel(), {"kind": "test"}, "Use the map.")
    config = json.loads((tmp_path / "config.json").read_text())
    assert config == {"format_version": FORMAT_VERSION, "model": "scripted",
                      "run": {"kind": "test"},
                      "knowledge": {"file": "knowledge.txt",
                                    "sha256": hashlib.sha256(b"Use the map.").hexdigest()}}
    assert (tmp_path / "knowledge.txt").read_text() == "Use the map."


def test_turn_rows_carry_the_task_and_action(tmp_path):
    log, agent, session = setup(tmp_path, ScriptedModel(), "Use the map.")
    log.record_turn(*agent.act(session))
    [action] = turns(tmp_path)
    assert action["event"] == "action" and action["source"] == "agent"
    assert action["turn"] == 1 and action["model_ms"] >= 0
    assert action["task"] == "Find exit" and "knowledge" not in action
    assert action["action"] == {"type": "press", "button": "A"}
    assert {"time", "screenshot", "after", "index",
            "action_number", "tick_before", "tick_after"} <= action.keys()
    assert not {"memory", "response"} & action.keys()


def test_model_call_rows_carry_input_and_response_by_turn(tmp_path):
    model = ScriptedModel()
    response = {"choices": [{"message": {"content": "{}", "reasoning": "Thinking."}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 60,
                          "completion_tokens_details": {"reasoning_tokens": 20}}}
    model.last_response = response
    log, agent, session = setup(tmp_path, model)
    log.record_turn(*agent.act(session))
    log.record_turn(*agent.act(session))
    first, second = model_calls(tmp_path)
    assert [first["turn"], second["turn"]] == [1, 2]
    assert second["ms"] >= 0 and second["response"] == response
    assert second["internal_reasoning"] == "Thinking."
    assert second["tokens"] == {"prompt": 900, "output": 40, "reasoning": 20}
    unknown = {"position": None, "tile": None, "direction": None, "characters": [],
               "species": None}
    assert second["input"] == {
        "task": "Find exit", "scratchpad": [],
        "screenshot": "frames/000001.png", "game_state": unknown,
        "memory": [{"screenshot": "frames/000000.png", "game_state": unknown,
                    "action": {"type": "press", "button": "A"}, "reasoning": "Scripted."}]}


def test_agent_rows_carry_the_model_reasoning(tmp_path):
    model = ScriptedModel(reasoning="Door ahead.", edits=[ScratchpadEdit(add=("Find the door.",))],
                          findings=[[finding("Kecleon in Pokémon Square sells X-Ray Specs.")]])
    log, agent, session = setup(tmp_path, model)
    log.record_turn(*agent.act(session))
    log.record_manual(Press("B"), agent.manual(session, Press("B")))
    action, manual = turns(tmp_path)
    assert action["reasoning"] == "Door ahead."
    assert action["scratchpad"] == {"rewrite": None, "lines": [], "add": ["Find the door."]}
    assert action["scratchpad_problem"] is None
    assert action["findings"] == [{"kind": "dialogue",
                                   "text": "Kecleon in Pokémon Square sells X-Ray Specs."}]
    assert "reasoning" not in manual
    assert log.recent()[1]["reasoning"] == "Door ahead."  # the live view shows it


def test_stop_and_model_error_rows(tmp_path):
    model = ScriptedModel(stop=votes(True), outcome="stuck", reasoning="Blocked by a wall.")
    log, agent, session = setup(tmp_path, model)
    log.record_turn(*agent.act(session))
    model.error = RuntimeError("down")
    error, _ = agent.act(session)
    log.record_model_error(error)
    stop, error_row = turns(tmp_path)
    assert [stop["event"], error_row["event"]] == ["stop", "model_error"]
    assert "action" not in stop and stop["outcome"] == "stuck"
    assert stop["reasoning"] == "Blocked by a wall." and stop["findings"] == []
    assert stop["scratchpad"] == {"rewrite": None, "lines": [], "add": []}
    assert error_row["error"] == "RuntimeError: down" and error_row["task"] == "Find exit"
    assert [call["turn"] for call in model_calls(tmp_path)] == [1, 2]
    assert not log.recent()


def test_a_game_action_without_its_transition_is_rejected(tmp_path):
    log, agent, session = setup(tmp_path, ScriptedModel())
    turn, _ = agent.act(session)
    with pytest.raises(ValueError, match="transition"):
        log.record_turn(turn, None)


def test_recent_rows_are_newest_first_capped_and_cleared_by_reset(tmp_path):
    log, agent, session = setup(tmp_path, ScriptedModel())
    for _ in range(RECENT_ROWS + 2):
        log.record_manual(Press("B"), agent.manual(session, Press("B")))
    recent = log.recent()
    assert len(recent) == RECENT_ROWS
    assert recent[0]["action_number"] == RECENT_ROWS + 2
    assert recent[0]["source"] == "manual"
    log.record_reset("frames/000000.png")
    assert log.recent() == []
    log.record_run_stop("limit", None, 42, 42, "frames/000001.png")
    reset, run_stop = turns(tmp_path)[-2:]
    assert [reset["event"], run_stop["event"]] == ["reset", "run_stop"]
    assert {"ended": "limit", "outcome": None, "actions": 42, "limit": 42}.items() <= (
        run_stop.items())
