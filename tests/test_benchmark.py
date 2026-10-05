"""Check that the town goals can be reached and do not pass at their starts."""

import json
import sys
from unittest.mock import MagicMock, Mock

import pytest

from emulator import GbaEmulator
from mystery_dungeon.agent import DecisionScores
from mystery_dungeon import benchmark
from mystery_dungeon.benchmark import (CASES, CHECKPOINTS, CORE, ROM, Case, Grader,
                                       build_fixtures, press, restore, run_case)


@pytest.fixture(scope="module")
def town(tmp_path_factory):
    paths = (ROM, CORE, *(CHECKPOINTS / f"{name}.ss0" for name in
                          ("square", "face-npc", "npc-dialogue", "shop-interact")))
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and town checkpoints required")
    output = tmp_path_factory.mktemp("town")
    with GbaEmulator(ROM, CORE) as game:
        seeds = build_fixtures(game, output)
        yield game, seeds, output


ROUTES = {
    "square_down": ["DOWN"],
    "square_north_path": ["UP"] * 8,
    "square_approach_granbull": ["LEFT", "LEFT", "LEFT", "DOWN"],
    "square_talk_granbull": ["LEFT", "LEFT", "LEFT", "DOWN", "DOWN", "A"],
    "face_npc_talk": ["A"],
    "dialogue_finish": ["A"] * 9,
    "face_npc_finish": ["A"] * 16,
    "shop_talk": ["A"],
    "shop_friend_areas": ["A"] * 20,
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_case_is_reachable_by_scripted_play(town, case):
    game, seeds, output = town
    route = ROUTES[case.id]

    class Scripted:
        def __init__(self):
            self.index = 0

        def score(self, goal, guidance, frame, history, options):
            button = route[min(self.index, len(route) - 1)]
            self.index += 1
            return DecisionScores({name: float(name == button) for name in options},
                                  {"complete": 0.0, "incomplete": 1.0, "uncertain": 0.0})

    result = run_case(game, case, Scripted(), seeds, output / case.id)
    assert result["reached"], result


def test_standing_still_does_not_count_as_moving_down(town):
    game, seeds, _ = town
    restore(game, seeds["square"])
    case = next(case for case in CASES if case.id == "square_down")
    grader = Grader(case, game)
    for _ in range(3):
        press(game, "WAIT")
        assert not grader.observe("down")[0]


def test_moving_the_wrong_way_does_not_count(town):
    game, seeds, _ = town
    restore(game, seeds["square"])
    case = next(case for case in CASES if case.id == "square_down")
    grader = Grader(case, game)
    press(game, "RIGHT")
    assert not grader.observe("down")[0]


def test_shop_menu_requires_friend_area_selection(town):
    game, seeds, _ = town
    restore(game, seeds["shop-interact"])
    case = next(case for case in CASES if case.id == "shop_friend_areas")
    grader = Grader(case, game)
    press(game, "A")
    assert grader.observe("wigglytuff")[0]
    assert not grader.observe("shop_menu")[0]
    for _ in range(15):
        press(game, "A")
    assert grader.observe("shop_menu")[0]
    for _ in range(2):
        press(game, "A")
    assert not grader.observe("shop_menu")[0]


@pytest.mark.parametrize("case_id,route", [
    ("square_down", ["DOWN"]),
    ("square_north_path", ["UP"] * 8),
    ("square_approach_granbull", ["LEFT", "LEFT", "LEFT", "DOWN"]),
])
def test_movement_grader_tolerates_waiting(town, case_id, route):
    game, seeds, _ = town
    restore(game, seeds["square"])
    case = next(case for case in CASES if case.id == case_id)
    grader = Grader(case, game)
    for button in ["WAIT"] * 4 + route:
        press(game, button)
    assert grader.observe(case.criterion)[0]


def test_dialogue_reach_and_completion_are_separate(town):
    game, seeds, output = town
    case = next(case for case in CASES if case.id == "face_npc_talk")

    class Policy:
        def __init__(self, complete_at):
            self.calls = 0
            self.complete_at = complete_at

        def score(self, goal, guidance, frame, history, options):
            self.calls += 1
            done = self.calls >= self.complete_at
            return DecisionScores({name: float(name == "A") for name in options},
                                  {"complete": float(done), "incomplete": float(not done),
                                   "uncertain": 0.0})

    early = run_case(game, case, Policy(1), seeds, output / "premature_completion")
    assert early["false_completion"] and not early["reached"]
    finished = run_case(game, case, Policy(2), seeds, output / "correct_completion")
    assert finished["recognized"] and finished["steps"] == 1


def test_inference_finishing_after_deadline_cannot_act_or_complete(town, monkeypatch):
    game, seeds, output = town
    clock = Mock(return_value=0)

    class LatePolicy:
        def score(self, goal, guidance, frame, history, options):
            clock.return_value = 181
            return DecisionScores({name: float(name == "DOWN") for name in options},
                                  {"complete": 1.0, "incomplete": 0.0, "uncertain": 0.0})

    case = next(case for case in CASES if case.id == "square_down")
    with monkeypatch.context() as patcher:
        patcher.setattr(benchmark.time, "monotonic", clock)
        result = run_case(game, case, LatePolicy(), seeds, output / "late_decision")
    assert result["stop_reason"] == "time_limit"
    assert result["steps"] == 0
    assert not result["false_completion"]
    rows = [json.loads(line) for line in
            (output / "late_decision/decisions.jsonl").read_text().splitlines()]
    discarded = next(row for row in rows if row["event"] == "discarded_decision")
    assert discarded["reason"] == "time_limit"


def test_main_loads_a_fresh_model_for_each_case(tmp_path, monkeypatch):
    rom, core, model_file = (tmp_path / name for name in ("game.gba", "core", "model.gguf"))
    for path in (rom, core, model_file, *(tmp_path / f"{name}.ss0" for name in
                              ("square", "face-npc", "npc-dialogue", "shop-interact"))):
        path.write_bytes(b"test")
    models = [object(), object()]
    cases = (Case("one", "square", "Goal one", "down", 1),
             Case("two", "square", "Goal two", "down", 1))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--rom", str(rom), "--core", str(core),
                                  "--output", str(tmp_path / "run"),
                                  "--model-file", str(model_file),
                                  "--decision-model", "fake:Model"])
    monkeypatch.setattr(benchmark, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(benchmark, "CASES", cases)
    monkeypatch.setattr(benchmark, "GbaEmulator", MagicMock())
    monkeypatch.setattr(benchmark, "build_fixtures", Mock(return_value={}))
    load = Mock(side_effect=models)
    run = Mock(return_value={"stop_reason": "budget", "steps": 0})
    monkeypatch.setattr(benchmark, "load_decision_model", load)
    monkeypatch.setattr(benchmark, "run_case", run)
    monkeypatch.setattr(benchmark, "report", Mock())
    benchmark.main()
    assert load.call_count == 2
    assert run.call_args_list[0].args[2] is models[0]
    assert run.call_args_list[1].args[2] is models[1]
    assert run.call_args_list[0].args[4] == tmp_path / "run/cases/one"
    assert run.call_args_list[1].args[4] == tmp_path / "run/cases/two"
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["hashes"][str(model_file)] == benchmark.sha256(model_file)
