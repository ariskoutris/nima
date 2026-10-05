"""Benchmark grading and stops against the local checkpoint."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from emulator import GbaEmulator
from mystery_dungeon.agent import DecisionScores
from mystery_dungeon import benchmark
from mystery_dungeon.benchmark import (CASES, CORE, ROM, STATE, MAILBOX_ROUTE, Case, Grader,
                                       build_fixtures, read_ss0, restore, run_case)


@pytest.fixture(scope="module")
def benchmark_env(tmp_path_factory):
    if not all(path.is_file() for path in (ROM, CORE, STATE)):
        pytest.skip("local ROM, core, and mGBA checkpoint required")
    root = tmp_path_factory.mktemp("benchmark")
    source = read_ss0(STATE)
    with GbaEmulator(ROM, CORE) as game:
        seeds, refs = build_fixtures(game, source, root)
        yield SimpleNamespace(root=root, source=source, game=game, seeds=seeds, refs=refs)


def test_checkpoint_contains_state_and_save_ram(benchmark_env):
    env = benchmark_env
    assert len(env.source.save_ram) == 131072
    assert len(env.source.state) > 300000
    damaged = bytearray(STATE.read_bytes())
    damaged[100] ^= 1
    path = env.root / "damaged.ss0"
    path.write_bytes(damaged)
    with pytest.raises(ValueError):
        read_ss0(path)


def test_mailbox_side_and_npc_are_not_mailbox_success(benchmark_env):
    env = benchmark_env
    restore(env.game, env.source)
    ram = env.game.system_ram()
    grader = Grader(env.refs, ram)
    for name in ("outside", "mailbox_side", "npc"):
        assert not grader.observe(env.refs[name], ram, "mailbox")[0], name
    assert grader.observe(env.refs["mailbox"], ram, "mailbox")[0]
    assert grader.observe(env.refs["cave"], ram, "mailbox_cave")[0]


def test_reached_recognized_and_false_completion(benchmark_env):
    env = benchmark_env

    class Policy:
        def __init__(self, button, complete_after=None):
            self.button = button
            self.complete_after = complete_after
            self.calls = 0

        def score(self, goal, guidance, frame, history, options):
            self.calls += 1
            done = self.complete_after is not None and self.calls >= self.complete_after
            return DecisionScores({name: float(name == self.button) for name in options},
                                  {"complete": float(done), "incomplete": float(not done),
                                   "uncertain": 0.0})

    south = next(case for case in CASES if case.id == "room_south")
    reached = run_case(env.game, south, Policy("DOWN"), env.seeds, env.refs,
                       env.root / "south")
    assert reached["reached"]
    assert reached["missed"]
    assert not reached["recognized"]
    assert reached["decisions"] == 4  # One action to reach, then three recognition checks.

    recognized = run_case(env.game, south, Policy("DOWN", 2), env.seeds, env.refs,
                          env.root / "recognized")
    assert recognized["recognized"]
    assert recognized["steps"] == 1
    assert (env.root / "recognized/frames/final.png").is_file()
    assert (env.root / "recognized/states/final.state").is_file()
    config = json.loads((env.root / "recognized/config.json").read_text())
    assert config["run"]["case"] == "room_south"

    false = run_case(env.game, south, Policy("DOWN", 1), env.seeds, env.refs,
                     env.root / "false")
    assert false["false_completion"]
    assert not false["reached"]
    assert false["steps"] == 0

    blocked = Case("wait_loop", "outside", "Find the clearing", "clearing", 16)
    result = run_case(env.game, blocked, Policy("WAIT"), env.seeds, env.refs,
                      env.root / "wait_loop")
    assert result["stop_reason"] in ("loop", "stalled", "budget")
    assert not result["reached"]
    if result["stop_reason"] in ("loop", "stalled"):
        rows = [json.loads(line) for line in (env.root / "wait_loop/decisions.jsonl").read_text().splitlines()]
        discarded = [row for row in rows if row["event"] == "discarded_decision"]
        assert discarded[-1]["reason"] == result["stop_reason"]
        wait = next(row for row in rows if row["event"] == "action" and row["source"] == "transition_wait")
        assert discarded[-1]["screenshot"] == wait["before"]


def test_inference_finishing_after_deadline_cannot_act_or_complete(benchmark_env, monkeypatch):
    env = benchmark_env
    clock = Mock(return_value=0)

    class LatePolicy:
        def score(self, goal, guidance, frame, history, options):
            clock.return_value = 181
            return DecisionScores({name: float(name == "DOWN") for name in options},
                                  {"complete": 1.0, "incomplete": 0.0, "uncertain": 0.0})

    case = next(case for case in CASES if case.id == "room_south")
    with monkeypatch.context() as patcher:
        patcher.setattr(benchmark.time, "monotonic", clock)
        result = run_case(env.game, case, LatePolicy(), env.seeds, env.refs,
                          env.root / "late_decision")
    assert result["stop_reason"] == "time_limit"
    assert result["steps"] == 0
    assert not result["false_completion"]
    rows = [json.loads(line) for line in (env.root / "late_decision/decisions.jsonl").read_text().splitlines()]
    discarded = next(row for row in rows if row["event"] == "discarded_decision")
    assert discarded["reason"] == "time_limit"
    assert discarded["completion_scores"]["complete"] == 1.0


ROUTES = {
    "room_south": ["DOWN"],
    "room_exit_directional": ["DOWN"] * 8,
    "room_exit_goal": ["DOWN"] * 8,
    "outside_talk": ["DOWN", "A"],
    "npc_ready_talk": ["A"],
    "outside_mailbox": list(MAILBOX_ROUTE),
    "mailbox_front_check": ["A"],
    "mailbox_front_button": ["A"],
    "room_then_talk": ["DOWN"] * 9 + ["A"],
    "outside_clearing": ["RIGHT"] * 7 + ["UP"] * 10,
    "bridge_to_clearing": ["UP"] * 10,
    "bridge_up_literal": ["UP"] * 10,
    "mailbox_to_cave": ["A"] * 120,
    "outside_mailbox_to_cave": list(MAILBOX_ROUTE) + ["A"] * 140,
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_every_case_has_a_successful_reference_route(benchmark_env, case):
    env = benchmark_env
    route = ROUTES[case.id]

    class Scripted:
        def __init__(self):
            self.index = 0

        def score(self, goal, guidance, frame, history, options):
            button = route[min(self.index, len(route) - 1)]
            self.index += 1
            return DecisionScores({name: float(name == button) for name in options},
                                  {"complete": 0.0, "incomplete": 1.0, "uncertain": 0.0})

    result = run_case(env.game, case, Scripted(), env.seeds, env.refs,
                      env.root / f"reference_{case.id}")
    assert result["reached"], result


def test_main_loads_a_fresh_model_for_each_case(tmp_path, monkeypatch):
    files = [tmp_path / name for name in ("game.gba", "core", "start.ss0")]
    for path in files:
        path.write_bytes(b"test")
    models = [object(), object()]
    cases = (Case("one", "inside", "Goal one", "south", 1),
             Case("two", "inside", "Goal two", "south", 1))
    args = ["benchmark", str(files[0]), "--core", str(files[1]),
            "--state", str(files[2]), "--output", str(tmp_path / "run"),
            "--decision-model", "fake:Model"]
    load = Mock(side_effect=models)
    run = Mock(return_value={"stop_reason": "budget", "steps": 0})
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(benchmark, "CASES", cases)
    monkeypatch.setattr(benchmark, "read_ss0", Mock())
    monkeypatch.setattr(benchmark, "GbaEmulator", MagicMock())
    monkeypatch.setattr(benchmark, "build_fixtures", Mock(return_value=({}, {})))
    monkeypatch.setattr(benchmark, "load_decision_model", load)
    monkeypatch.setattr(benchmark, "run_case", run)
    monkeypatch.setattr(benchmark, "report", Mock())
    benchmark.main()
    assert load.call_count == 2
    assert run.call_args_list[0].args[2] is models[0]
    assert run.call_args_list[1].args[2] is models[1]
    assert run.call_args_list[0].args[5] == tmp_path / "run/cases/one"
    assert run.call_args_list[1].args[5] == tmp_path / "run/cases/two"
