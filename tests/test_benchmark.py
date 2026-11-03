"""Check that the town tasks can be reached and do not pass at their starts."""

import json
import sys
from unittest.mock import MagicMock, Mock

import pytest
from fakes import ScriptedModel, press, rows, votes

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
from mystery_dungeon.model import ChatSettings
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
    "square_charizard_then_granbull": ["LEFT"] * 4 + ["UP"] * 3 + ["A"] * 3 + ["DOWN"] * 5
    + ["A"],
    "square_talk_five": ["LEFT"] * 4 + ["UP"] * 3 + ["A"] * 3 + ["RIGHT", "RIGHT", "UP"]
    + ["A"] * 3 + ["RIGHT"] * 4 + ["DOWN"] + ["A"] * 4 + ["RIGHT"] + ["DOWN"] * 4 + ["A"] * 9
    + ["LEFT"] * 6 + ["DOWN", "A"],
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

    def stop():
        earlier_calls = sum(call[0] == "call" for call in model.calls) - 1
        return case.criterion in ("finish", "finish_lombre") and earlier_calls == len(route)

    model = ScriptedModel(route, stop)
    result = run_case(game, case, model, seeds, output / case.id)
    assert result["reached"], result
    if case.criterion in ("finish", "finish_lombre"):
        assert result["stopped_reached"], result


@pytest.mark.parametrize("criterion,route", [
    ("charizard_granbull", ROUTES["square_talk_granbull"]),  # Granbull without Charizard
    ("talk_five", ROUTES["square_talk_five"][:-2]),  # all but the last of the five
])
def test_multi_talk_criteria_need_every_talk(town, criterion, route):
    game, seeds, output = town
    case = Case(f"partial_{criterion}", "square", "Talk", criterion, len(route))
    result = run_case(game, case, ScriptedModel(route), seeds, output / case.id)
    assert result["ended"] == "limit" and not result["reached"]


def test_lombre_finish_requires_closed_dialogue(town, tmp_path):
    game, seeds, _ = town
    restore(game, seeds["face-lombre"])
    case = next(case for case in CASES if case.id == "face_lombre_finish")
    session = GameplaySession(game, tmp_path)
    grader = Grader(case, session.ram)
    session.observe()
    for index in range(9):
        session.apply(Press("A"))
        session.observe()
        assert grader.reached() == (index == 8)
    session.apply(Press("A"))
    session.observe()
    assert not grader.reached()


def test_reopening_granbull_dialogue_does_not_count_as_finished(town):
    game, seeds, output = town
    case = Case("reopened_granbull", "face-npc", "Finish talking to Granbull", "finish", 14)

    result = run_case(game, case, ScriptedModel(["A"]), seeds, output / case.id)
    assert result["actions"] == 14
    assert result["ended"] == "limit" and result["limit"] == 14
    assert result["outcome"] is None
    assert not result["reached"]
    assert not result["stopped_reached"]
    log = rows(output / case.id / "turns.jsonl")
    assert [row["event"] for row in log].count("action") == 14  # no call past the limit
    assert {"ended": "limit", "outcome": None, "actions": 14, "limit": 14}.items() <= (
        log[-1].items())
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


def test_dialogue_reach_and_stop_are_separate(town):
    game, seeds, output = town
    case = next(case for case in CASES if case.id == "face_npc_talk")

    early = run_case(game, case, ScriptedModel(["A"], votes(True), outcome="stuck"),
                     seeds, output / "premature_stop")
    assert early["stopped_unreached"] and not early["reached"]
    assert early["ended"] == "stop" and early["outcome"] == "stuck"
    assert {"ended": "stop", "outcome": "stuck"}.items() <= (
        rows(output / "premature_stop/turns.jsonl")[-1].items())
    finished = run_case(game, case, ScriptedModel(["A"], votes(False, True)), seeds,
                        output / "correct_stop")
    assert finished["stopped_reached"] and finished["actions"] == 1


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
    assert result["ended"] == "limit"
    assert result["reached"] and result["missed"]
    assert result["actions"] == result["turns"] == 1
    assert result["seconds"] == 181


def test_main_runs_every_case_with_one_model(tmp_path, monkeypatch):
    rom, core = tmp_path / "game.gba", tmp_path / "core"
    for path in (rom, core, *(tmp_path / f"{name}.ss0" for name in benchmark.CHECKPOINT_NAMES)):
        path.write_bytes(b"test")
    cases = (Case("one", "square", "Task one", "down", 1),
             Case("two", "square", "Task two", "down", 1))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--rom", str(rom), "--core", str(core),
                                      "--output", str(tmp_path / "run")])
    model = object()
    settings = ChatSettings("http://endpoint", "vendor/model")
    monkeypatch.setattr(benchmark, "read_settings", Mock(return_value=settings))
    monkeypatch.setattr(benchmark, "ChatModel", Mock(return_value=model))
    monkeypatch.setattr(benchmark, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(benchmark, "CASES", cases)
    monkeypatch.setattr(benchmark, "GbaEmulator", MagicMock())
    monkeypatch.setattr(benchmark, "check_rom", Mock())
    monkeypatch.setattr(benchmark, "build_fixtures", Mock(return_value={}))
    run = Mock(return_value={"ended": "limit", "actions": 0})
    monkeypatch.setattr(benchmark, "run_case", run)
    monkeypatch.setattr(benchmark, "report", Mock())
    benchmark.main()
    assert [call.args[2] for call in run.call_args_list] == [model, model]
    assert run.call_args_list[0].args[4] == tmp_path / "run/cases/one"
    assert run.call_args_list[1].args[4] == tmp_path / "run/cases/two"
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["format_version"] == FORMAT_VERSION
    assert config["model"]["name"] == "vendor/model"
    assert config["memory_turns"] == benchmark.MEMORY_TURNS and config["instructions_sha256"]
    assert config["hashes"][str(rom)] == benchmark.sha256(rom)


def test_benchmark_loads_the_configured_knowledge_unless_given_another(tmp_path, monkeypatch):
    rom, core, knowledge = tmp_path / "game.gba", tmp_path / "core", tmp_path / "facts.md"
    knowledge.write_text("A talks.")
    for path in (rom, core, *(tmp_path / f"{name}.ss0" for name in benchmark.CHECKPOINT_NAMES)):
        path.write_bytes(b"test")
    monkeypatch.setattr(benchmark, "CHECKPOINTS", tmp_path)
    monkeypatch.setattr(benchmark, "CASES", (Case("one", "square", "Move", "down", 1),))
    monkeypatch.setattr(benchmark, "GbaEmulator", MagicMock())
    monkeypatch.setattr(benchmark, "check_rom", Mock())
    monkeypatch.setattr(benchmark, "build_fixtures", Mock(return_value={}))
    monkeypatch.setattr(benchmark, "ChatModel", Mock(return_value=object()))
    run = Mock(return_value={"ended": "limit", "actions": 0})
    monkeypatch.setattr(benchmark, "run_case", run)
    monkeypatch.setattr(benchmark, "report", Mock())
    for name, extra, path in (("default", [], benchmark.KNOWLEDGE),
                              ("other", ["--knowledge", str(knowledge)], knowledge)):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["benchmark", "--rom", str(rom), "--core", str(core),
                                      "--output", str(output), *extra])
        benchmark.main()
        config = json.loads((output / "config.json").read_text())
        assert config["knowledge"] == str(path) and str(path) in config["hashes"]
        assert run.call_args.args[5] == path.read_text()
