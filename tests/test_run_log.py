"""Run configuration and event rows."""

import hashlib
import json

from fakes import ScriptedModel, events, start, votes

from mystery_dungeon.actions import Press
from mystery_dungeon.agent import Agent
from mystery_dungeon.run_log import FORMAT_VERSION, RECENT_ROWS, RunLog


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


def test_turn_rows_carry_the_task_and_model_calls(tmp_path):
    model = ScriptedModel()
    model.last_response = {"reasoning": "door ahead"}
    log, agent, session = setup(tmp_path, model, "Use the map.")
    log.record_turn(*agent.act(session))
    [action] = events(tmp_path / "run")
    assert action["event"] == "action" and action["source"] == "agent"
    assert action["task"] == "Find exit" and "knowledge" not in action
    assert action["action"] == {"type": "press", "button": "A"}
    assert len(action["model_calls"]) == 1
    assert all(call["ms"] >= 0 and call["response"] == {"reasoning": "door ahead"}
               for call in action["model_calls"])
    assert {"time", "task", "screenshot", "after", "memory", "index",
            "action_number", "frame_before", "frame_after"} <= action.keys()
    log.record_turn(*agent.act(session))
    assert events(tmp_path / "run")[-1]["memory"] == [{
        "screenshot": "frames/000000.png", "action": {"type": "press", "button": "A"},
        "reasoning": "Scripted."}]


def test_agent_rows_carry_the_model_reasoning(tmp_path):
    log, agent, session = setup(tmp_path, ScriptedModel(reasoning="Door ahead."))
    log.record_turn(*agent.act(session))
    log.record_manual(Press("B"), agent.manual(session, Press("B")))
    action, manual = events(tmp_path / "run")
    assert action["reasoning"] == "Door ahead."
    assert "reasoning" not in manual


def test_completion_and_model_error_rows(tmp_path):
    model = ScriptedModel(complete=votes(True), reasoning="Done talking.")
    log, agent, session = setup(tmp_path, model)
    log.record_turn(*agent.act(session))
    model.error = RuntimeError("down")
    log.record_turn(*agent.act(session))
    rows = events(tmp_path / "run")
    assert [row["event"] for row in rows] == ["completion", "model_error"]
    assert "action" not in rows[0] and rows[0]["reasoning"] == "Done talking."
    assert rows[1]["error"] == "RuntimeError: down" and rows[1]["task"] == "Find exit"
    assert not log.recent()


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
    log.record_run_stop("turn_limit", "frames/000001.png")
    names = [row["event"] for row in events(tmp_path / "run")[-2:]]
    assert names == ["reset", "run_stop"]
