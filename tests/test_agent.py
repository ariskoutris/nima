"""Agent acting and memory through its public API."""

from dataclasses import fields

import pytest
from fakes import Game, ScriptedModel, start, votes

from mystery_dungeon.actions import Press, Stop, Wait
from mystery_dungeon.agent import (
    MEMORY_TURNS,
    Agent,
    Context,
    ModelError,
    Turn,
)
from mystery_dungeon.session import GameState


def agent_with(model, task="Find exit", knowledge=""):
    agent = Agent(model, knowledge)
    agent.set_task(task)
    return agent


def test_turns_act_on_the_task_knowledge_and_memory(tmp_path):
    model = ScriptedModel()
    agent = agent_with(model, task="  Find exit ", knowledge="  Use the map.  ")
    session = start(tmp_path)
    agent.act(session)
    turn, _ = agent.act(session)
    assert turn.context == Context("Find exit", "Use the map.", turn.context.memory,
                                   GameState(None, None, None))
    assert [entry.action for entry in turn.context.memory] == [Press("A")]
    assert model.calls[-1] == ("call", turn.context, turn.observation.frame)


def test_a_new_task_clears_memory(tmp_path):
    agent = agent_with(ScriptedModel())
    session = start(tmp_path)
    agent.act(session)
    agent.set_task("Find exit")
    assert len(agent.memory) == 1  # the same task keeps its turns
    agent.set_task("Talk to Granbull")
    assert agent.memory == ()


def test_the_plan_is_replaced_and_notes_are_kept_in_order(tmp_path):
    model = ScriptedModel(["A"], plans=["Find Kecleon, then buy a berry.", "",
                                        "Buy a berry from Kecleon."],
                          notes=["Done: found Kecleon at the shop.", None,
                                 "Failed: A facing away from Kecleon opened nothing."])
    agent = agent_with(model)
    session = start(tmp_path)
    for _ in range(4):
        agent.act(session)
    assert (model.calls[0][1].plan, model.calls[0][1].notes) == ("", ())
    assert model.calls[2][1].plan == "Find Kecleon, then buy a berry."  # blank kept it
    assert model.calls[3][1].plan == agent.plan == "Buy a berry from Kecleon."
    assert agent.notes == ("Done: found Kecleon at the shop.",
                           "Failed: A facing away from Kecleon opened nothing.")


def test_a_note_written_again_word_for_word_is_kept_once(tmp_path):
    agent = agent_with(ScriptedModel(notes=["Done: talked to Kecleon."] * 2))
    session = start(tmp_path)
    agent.act(session)
    agent.act(session)
    assert agent.notes == ("Done: talked to Kecleon.",)


def test_a_new_task_or_cleared_memory_forgets_plan_and_notes(tmp_path):
    model = ScriptedModel(["A"], plans=["Leave."] * 3, notes=["A line."] * 3)
    agent = agent_with(model)
    session = start(tmp_path)
    agent.act(session)
    agent.set_task("Find exit")
    assert (agent.plan, agent.notes) == ("Leave.", ("A line.",))  # the same task keeps them
    agent.set_task("Talk to Granbull")
    assert (agent.plan, agent.notes) == ("", ())
    agent.act(session)
    agent.clear_memory()
    assert (agent.plan, agent.notes) == ("", ())


def test_a_stop_applies_its_plan_and_note(tmp_path):
    model = ScriptedModel(stop=votes(True), plans=["Nothing left."],
                          notes=["Done: talked to Kecleon."])
    agent = agent_with(model)
    turn, transition = agent.act(start(tmp_path))
    assert transition is None and (turn.plan, turn.note) == ("Nothing left.",
                                                              "Done: talked to Kecleon.")
    assert (agent.plan, agent.notes) == ("Nothing left.", ("Done: talked to Kecleon.",))


def test_the_task_keeps_its_turns_for_the_review(tmp_path):
    agent = agent_with(ScriptedModel())
    session = start(tmp_path)
    turns = [agent.act(session)[0] for _ in range(MEMORY_TURNS + 1)]
    assert agent.turns == tuple(turns)  # more than memory keeps
    agent.set_task("Talk to Granbull")
    assert agent.turns == ()
    agent.act(session)
    agent.clear_memory()
    assert agent.turns == ()
    agent.act(session)
    agent.clear_turns()
    assert agent.turns == () and len(agent.memory) == 1  # memory stays for the model


def test_a_model_error_is_no_turn_of_the_task(tmp_path):
    agent = agent_with(ScriptedModel(error=RuntimeError("model down")))
    agent.act(start(tmp_path))
    assert agent.turns == ()


def test_the_first_task_keeps_earlier_manual_actions(tmp_path):
    agent = Agent(ScriptedModel())
    session = start(tmp_path)
    agent.manual(session, Press("RIGHT"))
    agent.set_task("Find exit")
    assert [entry.action for entry in agent.memory] == [Press("RIGHT")]


def test_memory_keeps_the_latest_turns_with_their_reasoning(tmp_path):
    model = ScriptedModel(["UP", "DOWN"], reasoning="Head for the door.")
    agent = agent_with(model)
    session = start(tmp_path)
    for _ in range(MEMORY_TURNS + 2):
        turn, _ = agent.act(session)
    assert turn.reasoning == "Head for the door."
    assert len(agent.memory) == MEMORY_TURNS
    assert agent.memory[-1].observation.screenshot == f"frames/{MEMORY_TURNS + 1:06d}.png"
    assert all(entry.reasoning == "Head for the door." for entry in agent.memory)
    assert model.calls[-1][1].memory == turn.context.memory


def test_manual_actions_enter_memory_without_reasoning(tmp_path):
    agent = agent_with(ScriptedModel(reasoning="Press A."))
    session = start(tmp_path)
    agent.act(session)
    agent.manual(session, Press("B"))
    assert [(entry.action, entry.reasoning) for entry in agent.memory] == [
        (Press("A"), "Press A."), (Press("B"), None)]


def test_manual_actions_enter_memory_until_cleared(tmp_path):
    model = ScriptedModel()
    agent = agent_with(model)
    session = start(tmp_path)
    agent.manual(session, Press("RIGHT"))
    assert [entry.action for entry in agent.memory] == [Press("RIGHT")]
    assert agent.memory[0].observation.screenshot == "frames/000000.png"
    agent.act(session)
    assert model.calls[-1][1].memory[0].action == Press("RIGHT")
    agent.clear_memory()
    assert agent.memory == ()


def test_stop_applies_nothing(tmp_path):
    game = Game()
    model = ScriptedModel(stop=votes(True))
    turn, transition = agent_with(model).act(start(tmp_path, game))
    assert turn.action == Stop("complete")
    assert transition is None
    assert game.steps == []
    assert [call[0] for call in model.calls] == ["call"]


def test_incomplete_task_applies_the_chosen_action(tmp_path):
    game = Game()
    turn, transition = agent_with(ScriptedModel(["WAIT"])).act(start(tmp_path, game))
    assert turn.action == Wait()
    assert transition.action_number == 1
    assert game.steps == [(8, ()), (4, ())]


def test_model_failures_become_model_errors(tmp_path):
    model = ScriptedModel(error=RuntimeError("model down"))
    game = Game()
    error, transition = agent_with(model).act(start(tmp_path, game))
    assert isinstance(error, ModelError)
    assert error.message == "RuntimeError: model down"
    assert error.context.task == "Find exit"
    assert transition is None and game.steps == []


def test_each_turn_makes_one_model_call(tmp_path):
    model = ScriptedModel()
    agent_with(model).act(start(tmp_path))
    assert [call[0] for call in model.calls] == ["call"]


def test_a_stop_keeps_its_outcome_and_reasoning(tmp_path):
    model = ScriptedModel(stop=votes(True), outcome="stuck", reasoning="A wall blocks the exit.")
    turn, _ = agent_with(model).act(start(tmp_path))
    assert turn.action == Stop("stuck")
    assert turn.reasoning == "A wall blocks the exit."


def test_missing_task_fails_before_any_model_call(tmp_path):
    model = ScriptedModel()
    agent = Agent(model)
    with pytest.raises(ValueError, match="task"):
        agent.act(start(tmp_path))
    assert model.calls == []


def test_turn_holds_only_what_the_agent_produced():
    assert {field.name for field in fields(Turn)} == {
        "observation", "context", "action", "reasoning", "plan", "note"}


def test_agent_writes_no_files(tmp_path):
    agent_with(ScriptedModel()).act(start(tmp_path))
    assert {path.parent.name for path in tmp_path.rglob("*") if path.is_file()} == {"frames"}


def test_agent_only_needs_observe_and_apply(tmp_path):
    session = start(tmp_path)

    class SessionPort:
        def observe(self):
            return session.observe()

        def apply(self, action):
            return session.apply(action)

    assert agent_with(ScriptedModel()).act(SessionPort())[0].action == Press("A")

