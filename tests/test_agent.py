"""Agent acting and memory through its public API."""

from dataclasses import fields

import pytest
from fakes import Game, ScriptedModel, finding, start, votes

from mystery_dungeon.actions import Press, Stop, Wait
from mystery_dungeon.agent import (
    EDIT_REJECTED,
    MEMORY_TURNS,
    Agent,
    Context,
    ModelError,
    Turn,
)
from mystery_dungeon.model import LineChange, ScratchpadEdit


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
    assert turn.context == Context("Find exit", "Use the map.", turn.context.memory)
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


def test_scratchpad_edits_use_the_numbers_shown_this_turn(tmp_path):
    model = ScriptedModel(["A"], edits=[
        ScratchpadEdit(add=("Find Kecleon.", "Shop is east, a guess.", "Buy a berry.")),
        ScratchpadEdit(lines=(LineChange(1, None), LineChange(2, "Shop is east of the well.")),
                       add=("Talked to Kecleon.",)),
        ScratchpadEdit(lines=(LineChange(1, None), LineChange(3, None))),
        ScratchpadEdit(rewrite=("Bought a berry.", "Task done."))])
    agent = agent_with(model)
    session = start(tmp_path)
    for _ in range(4):
        agent.act(session)
    assert model.calls[0][1].scratchpad == ()
    assert model.calls[1][1].scratchpad == ("Find Kecleon.", "Shop is east, a guess.",
                                            "Buy a berry.")
    # Changed in place, the deletion renumbers, and the addition goes at the end.
    assert model.calls[2][1].scratchpad == ("Shop is east of the well.", "Buy a berry.",
                                            "Talked to Kecleon.")
    assert model.calls[3][1].scratchpad == ("Buy a berry.",)
    assert agent.scratchpad == ("Bought a berry.", "Task done.")


@pytest.mark.parametrize("edit,problem", [
    (ScratchpadEdit(lines=(LineChange(7, None),)), "lines [7] are not shown"),
    (ScratchpadEdit(lines=(LineChange(1, None), LineChange(1, "Twice."))),
     "lines [1] were changed more than once"),
    (ScratchpadEdit(rewrite=("Fresh.",), add=("More.",)), "a rewrite came with"),
    (ScratchpadEdit(add=(" ",)), "a line was empty"),
    (ScratchpadEdit(rewrite=("[1] Talked to Kecleon.",)), "started with a line number"),
])
def test_a_rejected_edit_keeps_the_turn_and_says_why_in_the_scratchpad(tmp_path, edit, problem):
    game = Game()
    model = ScriptedModel(["A"], edits=[ScratchpadEdit(add=("Find Kecleon.",)), edit],
                          findings=[[], [finding("Kecleon runs the shop in Pokémon Square.")]])
    agent = agent_with(model)
    session = start(tmp_path, game)
    agent.act(session)
    turn, transition = agent.act(session)
    assert problem in turn.scratchpad_problem
    assert transition is not None and len(agent.findings) == 1  # the action and findings apply
    first, note = agent.scratchpad
    assert first == "Find Kecleon." and note == EDIT_REJECTED.format(
        problem=turn.scratchpad_problem)


def test_an_edit_that_applies_has_no_problem(tmp_path):
    turn, _ = agent_with(ScriptedModel(edits=[ScratchpadEdit(add=("Find Kecleon.",))])).act(
        start(tmp_path))
    assert turn.scratchpad_problem is None


def test_findings_accumulate_without_reaching_the_model(tmp_path):
    first, second = finding("Charizard in Pokémon Square says its team is Gold Rank."), finding(
        "Granbull stands south of the plaza in Pokémon Square.", kind="location")
    model = ScriptedModel(["A"], findings=[[first], [], [second]])
    agent = agent_with(model)
    session = start(tmp_path)
    for _ in range(3):
        turn, _ = agent.act(session)
    assert turn.findings == (second,)
    assert agent.findings == (first, second)
    assert "findings" not in {field.name for field in fields(Context)}


def test_a_new_task_or_cleared_memory_forgets_scratchpad_and_findings(tmp_path):
    model = ScriptedModel(["A"], edits=[ScratchpadEdit(add=("A line.",))] * 3,
                          findings=[[finding("A fact.")]] * 3)
    agent = agent_with(model)
    session = start(tmp_path)
    agent.act(session)
    agent.set_task("Find exit")
    assert len(agent.scratchpad) == len(agent.findings) == 1  # the same task keeps them
    agent.set_task("Talk to Granbull")
    assert agent.scratchpad == () and agent.findings == ()
    agent.act(session)
    assert agent.scratchpad == ("A line.",)
    agent.clear_memory()
    assert agent.scratchpad == () and agent.findings == ()


def test_a_stop_applies_its_scratchpad_edit_and_findings(tmp_path):
    model = ScriptedModel(stop=votes(True), edits=[ScratchpadEdit(add=("Talked to Kecleon.",))],
                          findings=[[finding("Kecleon runs the shop in Pokémon Square.")]])
    agent = agent_with(model)
    agent.act(start(tmp_path))
    assert agent.scratchpad == ("Talked to Kecleon.",)
    assert len(agent.findings) == 1


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
        "observation", "context", "action", "reasoning", "scratchpad", "findings",
        "scratchpad_problem"}


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

