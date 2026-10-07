"""Agent planning, acting and memory through its public API."""

from dataclasses import fields

import pytest

from fakes import Game, ScriptedModel, start, votes
from mystery_dungeon.actions import Press, Wait
from mystery_dungeon.agent import Agent, Turn


def agent_with(model, instructions="Find exit", knowledge=""):
    agent = Agent(model, model, knowledge)
    agent.set_instructions(instructions)
    return agent


def test_new_instructions_start_a_new_task_and_keep_memory(tmp_path):
    model = ScriptedModel()
    agent = agent_with(model, knowledge="  Use the map.  ")
    session = start(tmp_path)
    agent.take_turn(session)
    assert agent.task.description == "Find exit"
    assert agent.task.knowledge == "Use the map."
    agent.set_instructions("Talk to Granbull")
    turn, _ = agent.take_turn(session)
    assert turn.task.description == "Talk to Granbull"
    assert turn.task.knowledge == "Use the map."
    assert [entry.action for entry in turn.memory] == [Press("A")]


def test_manual_actions_enter_memory_until_cleared(tmp_path):
    model = ScriptedModel()
    agent = agent_with(model)
    session = start(tmp_path)
    agent.manual(session, Press("RIGHT"))
    assert [entry.action for entry in agent.memory] == [Press("RIGHT")]
    assert agent.memory[0].screenshot == "frames/000000.png"
    agent.take_turn(session)
    assert model.calls[-1][4][0].action == Press("RIGHT")
    agent.clear_memory()
    assert agent.memory == ()


def test_completion_applies_nothing(tmp_path):
    game = Game()
    model = ScriptedModel(complete=votes(True))
    turn, transition = agent_with(model).take_turn(start(tmp_path, game))
    assert turn.status == "completion"
    assert transition is None
    assert game.steps == []
    assert [call[0] for call in model.calls] == ["is_complete"]


def test_incomplete_task_applies_the_chosen_action(tmp_path):
    game = Game()
    turn, transition = agent_with(ScriptedModel(["WAIT"])).take_turn(start(tmp_path, game))
    assert turn.status == "action"
    assert turn.action == Wait()
    assert transition.action_number == 1
    assert game.steps == [(8, ()), (4, ())]


@pytest.mark.parametrize("failing", ["is_complete", "choose_action"])
def test_model_failures_become_model_errors(tmp_path, failing):
    model = ScriptedModel()

    def fail(*args):
        raise RuntimeError("model down")

    setattr(model, failing, fail)
    game = Game()
    turn, transition = agent_with(model).take_turn(start(tmp_path, game))
    assert turn.status == "model_error"
    assert turn.error == "RuntimeError: model down"
    assert turn.task.description == "Find exit"
    assert transition is None and game.steps == []


def test_propose_never_touches_the_game(tmp_path):
    game = Game()
    session = start(tmp_path, game)
    turn = agent_with(ScriptedModel()).propose(session.observe())
    assert turn.action == Press("A")
    assert game.steps == []


def test_only_the_action_model_receives_knowledge(tmp_path):
    model = ScriptedModel()
    agent_with(model, knowledge="Doors are at the top.").take_turn(start(tmp_path))
    assert model.calls[0][0] == "is_complete" and "Doors are at the top." not in model.calls[0]
    assert model.calls[1][:3] == ("choose_action", "Find exit", "Doors are at the top.")


def test_missing_instructions_fail_before_any_model_call(tmp_path):
    model = ScriptedModel()
    agent = Agent(model, model)
    with pytest.raises(ValueError, match="instructions"):
        agent.take_turn(start(tmp_path))
    assert model.calls == []


def test_turn_holds_only_what_the_agent_produced():
    assert {field.name for field in fields(Turn)} == {
        "observation", "task", "memory", "complete", "action", "error"}


def test_agent_writes_no_files(tmp_path):
    agent_with(ScriptedModel()).take_turn(start(tmp_path))
    assert {path.parent.name for path in tmp_path.rglob("*") if path.is_file()} == {"frames"}


def test_agent_only_needs_observe_and_apply(tmp_path):
    session = start(tmp_path)

    class SessionPort:
        def observe(self):
            return session.observe()

        def apply(self, action):
            return session.apply(action)

    assert agent_with(ScriptedModel()).take_turn(SessionPort())[0].status == "action"


def test_repeated_actions_warn_without_stopping(tmp_path):
    game = Game()
    model = ScriptedModel()
    agent = agent_with(model)
    session = start(tmp_path, game)
    for _ in range(12):
        assert agent.take_turn(session)[0].status == "action"
    assert game.steps == [(8, ("A",)), (4, ())] * 12
    assert session.warning == "Possible stall: little screen change after 12 A presses"
