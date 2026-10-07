"""Live operator input interrupts agent turns and runs in order."""

import json
import threading
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path

import pytest

from fakes import Game, ScriptedModel, events, votes
from mystery_dungeon import live
from mystery_dungeon.session import Checkpoint

AGENT_PRESS = (8, ("A",))


def wait_until(run, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = run.snapshot()
        if predicate(state):
            return state
        time.sleep(0.01)
    raise AssertionError(f"timed out; last state: {run.snapshot()}")


def idle(state):
    return state["status"] in ("paused", "error")


@pytest.fixture
def start_run(tmp_path, monkeypatch):
    """Start a LiveRun over a fake game; ``load`` replaces model loading, ``ready`` waits for it."""
    threads = []
    monkeypatch.setattr(live, "FRAME_SECONDS", 0)

    def start(model=None, *, models=None, load=None, ready=True, server=None, spec="fake:Model"):
        game = Game()
        state_file = tmp_path / "start.state"
        state_file.write_bytes(b"start")
        models = {spec: model, **(models or {})}
        monkeypatch.setattr(live, "load_model", load or models.__getitem__)
        monkeypatch.setattr(live, "GbaEmulator", lambda rom, core: game)
        run = live.LiveRun(tmp_path / "game.gba", tmp_path / "core", state_file,
                           tmp_path / "output", spec, 8, 4, model_server=server)
        thread = threading.Thread(target=run.run)
        threads.append((run, thread))
        thread.start()
        wait_until(run, lambda state: state["status"] != "loading" and
                   (not ready or not state["requested_model"]))
        return run, game

    yield start
    for run, thread in threads:
        run.stop()
        thread.join(3)
        assert not thread.is_alive()


def test_default_checkpoint_is_square():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "config.toml").read_text())
    assert config["game"]["state"] == "data/game/checkpoints/square.ss0"


def test_ss0_startup_and_reset_restore_save_ram(start_run, monkeypatch):
    monkeypatch.setattr(live, "read_checkpoint", lambda path: Checkpoint(b"ss0-state", b"new-ram"))
    run, game = start_run(ScriptedModel())
    assert (game.state, game.ram) == (b"ss0-state", b"new-ram")
    game.ram = b"changed-ram"
    run.control({"command": "reset"})
    wait_until(run, lambda state: state["reset_count"] == 1)
    assert (game.state, game.ram) == (b"ss0-state", b"new-ram")


def test_manual_action_enters_memory_and_reset_keeps_instructions(start_run):
    model = ScriptedModel()
    run, game = start_run(model)
    run.control({"instructions": "Find exit", "command": "manual", "action": "RIGHT"})
    wait_until(run, lambda state: len(state["history"]) == 1)
    run.control({"command": "turn"})
    state = wait_until(run, lambda state: len(state["history"]) == 2 and idle(state))
    assert model.calls[-1][4][0].action.button == "RIGHT"
    assert [row["source"] for row in state["history"]] == ["agent", "manual"]
    assert (run.output / state["history"][1]["screenshot"]).is_file()
    assert (run.output / "states/latest.state").read_bytes() == game.state
    assert json.loads((run.output / "config.json").read_text())["run"]["kind"] == "live"
    run.control({"command": "reset"})
    state = wait_until(run, lambda state: state["reset_count"] == 1)
    assert state["instructions"] == "Find exit" and state["history"] == []
    assert state["notice"] == "Reset complete."
    run.control({"command": "turn"})
    state = wait_until(run, lambda state: len(state["history"]) == 1 and idle(state))
    assert state["history"][0]["action_number"] == 1
    assert model.calls[-1][4] == ()  # reset cleared memory


def test_stall_warning_shows_until_reset(start_run):
    run, _ = start_run(ScriptedModel())
    for _ in range(8):
        run.control({"command": "manual", "action": "DOWN"})
    state = wait_until(run, lambda state: len(state["history"]) == 8)
    assert state["warning"] == "Possible stall: little screen change after 8 DOWN presses"
    run.control({"command": "reset"})
    assert wait_until(run, lambda state: state["reset_count"] == 1)["warning"] is None


def test_upload_state_and_ss0(start_run, monkeypatch):
    run, game = start_run(ScriptedModel())
    run.load_state_upload(b"uploaded", "state")
    state = wait_until(run, lambda state: state["reset_count"] == 1)
    assert game.state == b"uploaded" and state["notice"] == "Loaded state."
    run.control({"command": "reset"})
    wait_until(run, lambda state: state["reset_count"] == 2)
    assert game.state == b"uploaded"

    monkeypatch.setattr(live, "read_ss0", lambda data: Checkpoint(b"ss0-state", b"new-ram"))
    run.load_state_upload(b"png-data", "ss0")
    wait_until(run, lambda state: state["reset_count"] == 3)
    assert (game.state, game.ram) == (b"ss0-state", b"new-ram")

    run.load_state_upload(b"bad", "state")
    state = wait_until(run, lambda state: state["status"] == "error")
    assert state["reset_count"] == 3 and game.state == b"ss0-state"
    run.control({"command": "reset"})
    wait_until(run, lambda state: state["reset_count"] == 4)
    with pytest.raises(ValueError, match="choose a"):
        run.load_state_upload(b"uploaded", "unknown")


@pytest.mark.parametrize("interrupt,reason,took_effect", [
    (lambda run: run.control({"command": "pause"}), "paused", lambda state: True),
    (lambda run: run.control({"command": "manual", "action": "B"}), "manual_control",
     lambda state: [row["action"]["button"] for row in state["history"]] == ["B"]),
    (lambda run: run.control({"command": "reset"}), "reset_requested",
     lambda state: state["reset_count"] == 1),
    (lambda run: run.load_state_upload(b"uploaded", "state"), "reset_requested",
     lambda state: state["reset_count"] == 1),
    (lambda run: run.request_model(live.GLM), "model_change",
     lambda state: state["model"] == live.GLM),
    (lambda run: run.control({"instructions": "Something else"}), "instructions_changed",
     lambda state: state["instructions"] == "Something else"),
])
def test_operator_input_discards_the_turn_in_progress(start_run, interrupt, reason, took_effect):
    gate = threading.Event()
    model = ScriptedModel(gate=gate)
    run, game = start_run(model, models={live.GLM: ScriptedModel()})
    run.control({"instructions": "Find exit", "command": "start"})
    assert model.entered.wait(2)
    interrupt(run)
    assert run.snapshot()["status"] == "cancelling"
    if reason == "instructions_changed":
        run.control({"command": "pause"})
    gate.set()
    wait_until(run, lambda state: any(row["event"] == "discarded" for row in events(run.output)))
    discarded = next(row for row in events(run.output) if row["event"] == "discarded")
    assert discarded["reason"] == reason and discarded["action"]["button"] == "A"
    wait_until(run, lambda state: idle(state) and took_effect(state))
    assert AGENT_PRESS not in game.steps


def test_pause_then_start_in_one_call_still_discards(start_run):
    gate = threading.Event()
    model = ScriptedModel(gate=gate)
    run, game = start_run(model)
    run.control({"instructions": "Find exit", "command": "start"})
    assert model.entered.wait(2)
    run.control({"command": "pause"})
    run.control({"command": "start"})
    gate.set()
    state = wait_until(run, lambda state: len(state["history"]) >= 1)
    run.control({"command": "pause"})
    names = [row["event"] for row in events(run.output)]
    assert names.index("discarded") < names.index("action")
    assert state["history"][-1]["source"] == "agent"


def test_one_turn_runs_exactly_one_turn(start_run):
    model = ScriptedModel()
    run, game = start_run(model)
    run.control({"instructions": "Find exit", "command": "turn"})
    wait_until(run, lambda state: len(state["history"]) == 1 and idle(state))
    time.sleep(0.05)
    assert len(run.snapshot()["history"]) == 1
    assert game.steps.count(AGENT_PRESS) == 1


def test_completion_pauses_without_pressing(start_run):
    run, game = start_run(ScriptedModel(complete=votes(True)))
    run.control({"instructions": "Done", "command": "start"})
    state = wait_until(run, lambda state: state["notice"] == "Completed: Done")
    assert state["status"] == "paused" and state["history"] == []
    assert AGENT_PRESS not in game.steps


def test_model_error_pauses_with_the_error(start_run):
    run, _ = start_run(ScriptedModel(error=RuntimeError("model down")))
    run.control({"instructions": "Find exit", "command": "start"})
    state = wait_until(run, lambda state: state["status"] == "error")
    assert state["error"] == "Model failed: RuntimeError: model down"
    run.control({"command": "pause"})
    assert run.snapshot()["status"] == "error"  # errors clear only on the next run, manual or reset


def test_manual_queued_behind_a_failed_first_model_keeps_its_error(start_run):
    gate = threading.Event()

    def load(spec):
        assert gate.wait(2)
        raise RuntimeError("no server")

    run, _ = start_run(load=load, ready=False)
    wait_until(run, lambda state: state["status"] == "switching")
    run.control({"command": "manual", "action": "A"})
    gate.set()
    wait_until(run, lambda state: state["status"] == "error" and not state["requested_model"])
    time.sleep(0.05)
    assert run.snapshot()["error"] == "Could not switch model: no server"


def test_control_validation(start_run):
    run, _ = start_run(ScriptedModel())
    live.handler_for(run)  # the static page resolves from its package location
    with pytest.raises(ValueError, match="instructions"):
        run.control({"command": "start"})
    with pytest.raises(ValueError, match="invalid manual action"):
        run.control({"command": "manual", "action": "START"})
    with pytest.raises(ValueError, match="unknown control field"):
        run.control({"goal": "Find exit"})
    with pytest.raises(ValueError, match="unknown model"):
        run.request_model("missing:Model")


def test_frames_stream_during_emulator_steps(start_run):
    run, game = start_run(ScriptedModel())
    before = run.wait_for_screen(None)[2]
    run.control({"command": "manual", "action": "WAIT"})
    wait_until(run, lambda state: len(state["history"]) == 1)
    screen, revision, frame = run.wait_for_screen(None)
    assert screen == game.frame.png_bytes() and revision >= 1
    assert frame == before + 12


def test_model_switch_manages_the_local_server(start_run, monkeypatch):
    class Server:
        def __init__(self):
            self.starts = self.stops = 0

        def start(self):
            self.starts += 1
            return "http://127.0.0.1:9999"

        def stop(self):
            self.stops += 1

    monkeypatch.setenv("QEV_BASE_URL", "")
    server = Server()
    run, _ = start_run(ScriptedModel(), models={live.GLM: ScriptedModel()},
                       server=server, spec=live.ONEJEV)
    assert server.starts == 1 and live.os.environ["QEV_BASE_URL"] == "http://127.0.0.1:9999"
    run.request_model(live.GLM)
    state = wait_until(run, lambda state: state["model"] == live.GLM)
    assert server.stops == 1
    assert state["notice"] == "Switched to GLM-5.3-Flash (OpenRouter)."
    assert [row["to"] for row in events(run.output) if row["event"] == "model_change"] == [live.GLM]


def test_model_server_owns_process_group(tmp_path, monkeypatch):
    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 9000)

    class Process:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout == 10

    @contextmanager
    def ready(url, timeout):
        assert url == "http://127.0.0.1:9000/v1/models"
        yield type("Response", (), {"status": 200})()

    calls = []
    monkeypatch.setattr(live.socket, "socket", Socket)
    monkeypatch.setattr(live.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs)) or Process())
    monkeypatch.setattr(live, "urlopen", ready)
    monkeypatch.setattr(live.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    server = live.ModelServer("model.gguf", "vision.gguf", tmp_path)
    assert server.start() == "http://127.0.0.1:9000"
    assert calls[0][0][1:4] == ["-m", "qev.cli", "serve"]
    assert calls[0][1]["start_new_session"] is True
    server.close()
    assert calls[-1] == (1234, live.signal.SIGTERM)
