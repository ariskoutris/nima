"""Check that the town tasks can be reached and do not pass at their starts."""

import json
import sys
from unittest.mock import MagicMock, Mock

import pytest
from fakes import ScriptedModel, press, votes

from emulator import GbaEmulator
from mystery_dungeon import benchmark
from mystery_dungeon.actions import Press
from mystery_dungeon.benchmark import (
    CASES,
    CHECKPOINT_NAMES,
    CHECKPOINTS,
    CORE,
    ROM,
    Case,
    Grader,
    build_fixtures,
    run_case,
)
from mystery_dungeon.ram_state import GameRam
from mystery_dungeon.run_log import FORMAT_VERSION
from mystery_dungeon.session import GameplaySession, read_ss0, restore


def test_ss0_can_be_read_from_upload_bytes():
    checkpoint = CHECKPOINTS / "square.ss0"
    if not checkpoint.is_file():
        pytest.skip("local mGBA checkpoint required")
    assert read_ss0(checkpoint.read_bytes()) == read_ss0(checkpoint)


@pytest.fixture(scope="module")
def town(tmp_path_factory):
    paths = (ROM, CORE, *(CHECKPOINTS / f"{name}.ss0" for name in CHECKPOINT_NAMES))
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and town checkpoints required")
    output = tmp_path_factory.mktemp("town")
    with GbaEmulator(ROM, CORE) as game:
        seeds = build_fixtures(game, output)
        yield game, seeds, output


ROUTES = {
    "square_down": ["DOWN"],
    "square_north_exit": ["UP"] * 20,
    "square_approach_granbull": ["LEFT", "LEFT", "LEFT", "DOWN"],
    "square_talk_granbull": ["LEFT", "LEFT", "LEFT", "DOWN", "DOWN", "A"],
    "face_npc_talk": ["A"],
    "face_lombre_finish": ["A"] * 9,
    "face_npc_talk_overlap": ["A"],
    "dialogue_finish": ["A"] * 2,
    "face_npc_finish": ["A"] * 3,
    "shop_talk": ["A"],
    "shop_buy_friend_area": ["A"] * 19,
    "shop_check_pidgey": ["A", "DOWN", "A", "A"] + ["DOWN"] * 4 + ["A"],
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_case_is_reachable_by_scripted_play(town, case):
    game, seeds, output = town
    route = ROUTES[case.id]

    def done():
        actions = sum(call[0] == "choose_action" for call in model.calls)
        return case.criterion in ("finish", "finish_lombre") and actions == len(route)

    model = ScriptedModel(route, done)
    result = run_case(game, case, model, seeds, output / case.id)
    assert result["reached"], result
    if case.criterion in ("finish", "finish_lombre"):
        assert result["recognized"], result


def test_lombre_finish_requires_closed_dialogue(town, tmp_path):
    game, seeds, _ = town
    restore(game, seeds["face-lombre"])
    case = next(case for case in CASES if case.id == "face_lombre_finish")
    session = GameplaySession(game, tmp_path)
    grader = Grader(case, session.ram)
    session.capture()
    for index in range(9):
        session.apply(Press("A"))
        session.wait_for_text()
        assert grader.reached() == (index == 8)
    session.apply(Press("A"))
    session.wait_for_text()
    assert not grader.reached()


def test_reopening_granbull_dialogue_does_not_count_as_finished(town):
    game, seeds, output = town
    case = Case("reopened_granbull", "face-npc", "Finish talking to Granbull", "finish", 14)

    result = run_case(game, case, ScriptedModel(["A"]), seeds, output / case.id)
    assert result["actions"] == 14
    assert result["stop_reason"] == "limit"
    assert not result["reached"]
    assert not result["recognized"]
    assert Grader(case, GameRam(game)).talking(235)


def test_standing_still_does_not_count_as_moving_down(town):
    game, seeds, _ = town
    restore(game, seeds["square"])
    case = next(case for case in CASES if case.id == "square_down")
    grader = Grader(case, GameRam(game))
    for _ in range(3):
        press(game, "WAIT")
        assert not grader.reached()


def test_moving_the_wrong_way_does_not_count(town):
    game, seeds, _ = town
    restore(game, seeds["square"])
    case = next(case for case in CASES if case.id == "square_down")
    grader = Grader(case, GameRam(game))
    press(game, "RIGHT")
    assert not grader.reached()


def test_north_exit_requires_leaving_pokemon_square(town):
    game, seeds, _ = town
    restore(game, seeds["square"])
    case = next(case for case in CASES if case.id == "square_north_exit")
    grader = Grader(case, GameRam(game))
    assert grader.ram.ground_map() == 1
    for _ in range(8):
        press(game, "UP")
    assert not grader.reached()
    for _ in range(12):
        press(game, "UP")
    assert grader.ram.ground_map() == 3
    assert grader.reached()


def test_buying_requires_choosing_a_friend_area(town):
    game, seeds, _ = town
    restore(game, seeds["shop-interact"])
    case = next(case for case in CASES if case.id == "shop_buy_friend_area")
    grader = Grader(case, GameRam(game))
    press(game, "A")
    assert Grader(Case("talk", "shop-interact", "", "wigglytuff", 1), GameRam(game)).reached()
    for _ in range(17):  # through the greeting and the Friend Areas list to Buy/Info
        press(game, "A")
        assert not grader.reached()
    press(game, "A")
    assert grader.reached()


def test_check_pidgey_requires_selecting_pidgey(town):
    game, seeds, _ = town
    case = next(case for case in CASES if case.id == "shop_check_pidgey")
    route_to_list = ["A"] * 8 + ["DOWN", "A"] + ["A"] * 6

    restore(game, seeds["shop-interact"])
    grader = Grader(case, GameRam(game))
    for button in route_to_list:
        press(game, button)
    assert not grader.reached()
    press(game, "A")  # Select the first entry, Squirtle.
    assert "Squirtle's Friend Area" in grader.ram.textbox_text()
    assert not grader.reached()

    restore(game, seeds["shop-interact"])
    grader = Grader(case, GameRam(game))
    for button in route_to_list + ["DOWN"] * 4:
        press(game, button)
    assert not grader.reached()
    press(game, "A")
    assert grader.reached()


@pytest.mark.parametrize("case_id,route", [
    ("square_down", ["DOWN"]),
    ("square_north_exit", ["UP"] * 20),
    ("square_approach_granbull", ["LEFT", "LEFT", "LEFT", "DOWN"]),
])
def test_movement_grader_tolerates_waiting(town, case_id, route):
    game, seeds, _ = town
    restore(game, seeds["square"])
    case = next(case for case in CASES if case.id == case_id)
    grader = Grader(case, GameRam(game))
    for button in ["WAIT"] * 4 + route:
        press(game, button)
    assert grader.reached()


def test_dialogue_reach_and_completion_are_separate(town):
    game, seeds, output = town
    case = next(case for case in CASES if case.id == "face_npc_talk")

    early = run_case(game, case, ScriptedModel(["A"], votes(True)), seeds,
                     output / "premature_completion")
    assert early["false_completion"] and not early["reached"]
    finished = run_case(game, case, ScriptedModel(["A"], votes(False, True)), seeds,
                        output / "correct_completion")
    assert finished["recognized"] and finished["actions"] == 1


def test_slow_inference_still_acts_but_action_limit_stops_after_reach(town, monkeypatch):
    game, seeds, output = town
    clock = Mock(return_value=0)

    def slow_check():
        clock.return_value = 181
        return False

    case = Case("slow_one_action", "square", "Move one tile down", "down", 1)
    with monkeypatch.context() as patcher:
        patcher.setattr(benchmark.time, "monotonic", clock)
        result = run_case(game, case, ScriptedModel(["DOWN"], slow_check), seeds,
                          output / "late_turn")
    assert result["stop_reason"] == "limit"
    assert result["reached"] and result["missed"]
    assert result["actions"] == result["turns"] == 1
    assert result["seconds"] == 181


def test_main_runs_every_case_with_one_model(tmp_path, monkeypatch):
    rom, core = tmp_path / "game.gba", tmp_path / "core"
    checkpoints = (tmp_path / f"{name}.ss0" for name in benchmark.CHECKPOINT_NAMES)
    for path in (rom, core, *checkpoints):
        path.write_bytes(b"test")
    model = object()
    cases = (Case("one", "square", "Task one", "down", 1),
             Case("two", "square", "Task two", "down", 1))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--rom", str(rom), "--core", str(core),
                                      "--output", str(tmp_path / "run")])
    monkeypatch.setattr(benchmark, "GlmModel", Mock(return_value=model))
    monkeypatch.setattr(benchmark, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(benchmark, "CASES", cases)
    monkeypatch.setattr(benchmark, "GbaEmulator", MagicMock())
    monkeypatch.setattr(benchmark, "build_fixtures", Mock(return_value={}))
    run = Mock(return_value={"stop_reason": "limit", "actions": 0})
    monkeypatch.setattr(benchmark, "run_case", run)
    monkeypatch.setattr(benchmark, "report", Mock())
    benchmark.main()
    assert [call.args[2] for call in run.call_args_list] == [model, model]
    assert run.call_args_list[0].args[4] == tmp_path / "run/cases/one"
    assert run.call_args_list[1].args[4] == tmp_path / "run/cases/two"
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["format_version"] == FORMAT_VERSION
    assert config["model"] == benchmark.MODEL
    assert "time_limit_seconds" not in config
    assert "recognition_window" not in config


def test_benchmark_knowledge_is_optional_and_recorded(tmp_path, monkeypatch):
    rom, core, knowledge = tmp_path / "game.gba", tmp_path / "core", tmp_path / "controls.md"
    knowledge.write_text("A talks.")
    for path in (rom, core, *(tmp_path / f"{name}.ss0" for name in benchmark.CHECKPOINT_NAMES)):
        path.write_bytes(b"test")
    monkeypatch.setattr(benchmark, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(benchmark, "CASES", (Case("one", "square", "Move", "down", 1),))
    monkeypatch.setattr(benchmark, "GbaEmulator", MagicMock())
    monkeypatch.setattr(benchmark, "build_fixtures", Mock(return_value={}))
    monkeypatch.setattr(benchmark, "GlmModel", Mock(return_value=object()))
    run = Mock(return_value={"stop_reason": "limit", "actions": 0})
    monkeypatch.setattr(benchmark, "run_case", run)
    monkeypatch.setattr(benchmark, "report", Mock())
    for name, extra in (("none", []), ("controls", ["--knowledge", str(knowledge)])):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["benchmark", "--rom", str(rom), "--core", str(core),
                                      "--output", str(output), *extra])
        benchmark.main()
        config = json.loads((output / "config.json").read_text())
        assert config["knowledge"] == (str(knowledge) if extra else None)
        assert run.call_args.args[5] == ("A talks." if extra else "")
