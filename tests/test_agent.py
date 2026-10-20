"""Agent acting and memory through its public API."""

from dataclasses import fields

import pytest
from fakes import PROGRESS, Game, ScriptedModel, note, start, votes

from mystery_dungeon.actions import Press, Stop, Wait
from mystery_dungeon.agent import MEMORY_TURNS, Agent, Context, ModelError, Turn
from mystery_dungeon.model import Progress


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
                                   progress=PROGRESS)
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


def test_notes_accumulate_and_stay_in_context_until_the_task_changes(tmp_path):
    berries, specs, closes = note("Sells Oran Berries."), note("Sells X-Ray Specs."), note(
        "Closes at night.", kind="mechanic")
    model = ScriptedModel(["A"], notes=[[berries], [], [specs, closes]])
    agent = agent_with(model)
    session = start(tmp_path)
    for _ in range(3):
        turn, _ = agent.act(session)
    assert turn.notes == (specs, closes)
    assert model.calls[-1][1].notes == (berries,)
    assert agent.notes == (berries, specs, closes)
    agent.set_task("Find exit")
    assert len(agent.notes) == 3  # the same task keeps its notes
    agent.set_task("Talk to Granbull")
    assert agent.notes == () and agent.progress is None


def test_only_the_latest_progress_reaches_the_next_turn(tmp_path):
    first, second = Progress((), ("Talk to Kecleon",)), Progress(("Talk to Kecleon",), ())
    model = ScriptedModel(["A"], progress=[first, second])
    agent = agent_with(model)
    session = start(tmp_path)
    turn, _ = agent.act(session)
    assert turn.context.progress is None and turn.progress == first
    agent.act(session)
    agent.act(session)
    assert [call[1].progress for call in model.calls] == [None, first, second]
    assert agent.progress == second


def test_a_stop_updates_progress_and_notes(tmp_path):
    done = Progress(("Talk to Kecleon",), ())
    model = ScriptedModel(stop=votes(True), progress=[done], notes=[[note("Sells Oran Berries.")]])
    agent = agent_with(model)
    agent.act(start(tmp_path))
    assert agent.progress == done and len(agent.notes) == 1


def test_clearing_memory_forgets_progress_and_notes(tmp_path):
    agent = agent_with(ScriptedModel(notes=[[note("A fact.")]]))
    agent.act(start(tmp_path))
    agent.clear_memory()
    assert agent.notes == () and agent.progress is None


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
        "observation", "context", "action", "reasoning", "progress", "notes"}


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


def test_repeated_actions_warn_without_stopping(tmp_path):
    game = Game()
    model = ScriptedModel()
    agent = agent_with(model)
    session = start(tmp_path, game)
    for _ in range(12):
        assert agent.act(session)[0].action == Press("A")
    assert game.steps == [(8, ("A",)), (4, ())] * 12
    assert session.warning == "Possible stall: little screen change after 12 A presses"
