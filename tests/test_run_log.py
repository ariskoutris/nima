"""Run configuration and event rows."""

import hashlib
import json

from fakes import ScriptedModel, events, start, votes

from mystery_dungeon.actions import Press
from mystery_dungeon.agent import Agent
from mystery_dungeon.run_log import FORMAT_VERSION, RECENT_ROWS, RunLog


def setup(tmp_path, model, knowledge=""):
    log = RunLog(tmp_path / "run", "fake:Model", {"kind": "test"}, knowledge)
    watched = log.watch(model)
    agent = Agent(watched, watched, knowledge)
    agent.set_instructions("Find exit")
    return log, agent, start(tmp_path / "run")


def test_config_snapshots_knowledge(tmp_path):
    RunLog(tmp_path, "fake:Model", {"kind": "test"}, "Use the map.")
    config = json.loads((tmp_path / "config.json").read_text())
    assert config == {"format_version": FORMAT_VERSION, "model": "fake:Model",
                      "run": {"kind": "test"},
                      "knowledge": {"file": "knowledge.txt",
                                    "sha256": hashlib.sha256(b"Use the map.").hexdigest()}}
    assert (tmp_path / "knowledge.txt").read_text() == "Use the map."


def test_turn_rows_carry_model_calls_and_task_changes(tmp_path):
    model = ScriptedModel()
    model.last_response = {"reasoning": "door ahead"}
    log, agent, session = setup(tmp_path, model, "Use the map.")
    log.record_turn(*agent.take_turn(session))
    task_changed, action = events(tmp_path / "run")
    assert task_changed["event"] == "task_changed" and task_changed["task"] == "Find exit"
    assert task_changed["knowledge"] == "Use the map."
    assert action["event"] == "action" and action["source"] == "agent"
    assert action["action"] == {"type": "press", "button": "A"}
    assert action["complete"] is False and "knowledge" not in action
    assert [call["call"] for call in action["model_calls"]] == ["is_complete", "choose_action"]
    assert all(call["ms"] >= 0 and call["response"] == {"reasoning": "door ahead"}
               for call in action["model_calls"])
    assert {"time", "task", "screenshot", "after", "memory", "model", "index",
            "action_number", "frame_before", "frame_after"} <= action.keys()
    log.record_turn(*agent.take_turn(session))
    assert events(tmp_path / "run")[-1]["event"] == "action"  # same task: no second task_changed
    assert events(tmp_path / "run")[-1]["memory"][0]["action"] == {"type": "press", "button": "A"}


def test_completion_error_and_discarded_rows(tmp_path):
    model = ScriptedModel(complete=votes(True))
    log, agent, session = setup(tmp_path, model)
    log.record_turn(*agent.take_turn(session))
    log.record_discarded(agent.propose(session.observe()), "paused")
    model.error = RuntimeError("down")
    log.record_turn(*agent.take_turn(session))
    rows = events(tmp_path / "run")[1:]
    assert [row["event"] for row in rows] == ["completion", "discarded", "model_error"]
    assert rows[0]["complete"] is True and "action" not in rows[0]
    assert rows[1]["reason"] == "paused" and rows[1]["action"]["button"] == "A"
    assert rows[2]["error"] == "RuntimeError: down"
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
