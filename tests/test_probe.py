"""Probe dispatch and reporting without running model inference."""

import json
from pathlib import Path

import pytest

from scripts import probe


def write_result(command, reason="recognized"):
    output = Path(command[command.index("--output") + 1])
    case = command[command.index("--case") + 1]
    folder = output / "cases" / case
    folder.mkdir(parents=True)
    row = {"case": case, "reached": reason == "recognized",
           "recognized": reason == "recognized", "missed": False,
           "false_completion": False, "stop_reason": reason,
           "actions": 1, "seconds": 2.0}
    (output / "results.json").write_text(json.dumps([row]))
    (folder / "events.jsonl").write_text(
        json.dumps({"event": "action", "source": "agent",
                    "model_calls": [{"ms": 40}, {"ms": 60}]}) + "\n" if reason != "model_error"
        else json.dumps({"event": "model_error"}) + "\n")


def test_glm_variants_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        write_result(command)

    monkeypatch.setattr(probe.subprocess, "run", run)
    output = tmp_path / "probe"
    probe.main(["--model", "glm", "--case", "square_down", "--variant", "baseline",
                "--variant", "controls", "--output", str(output)])
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index("--model") + 1] == "mystery_dungeon.models.glm:GlmModel"
    assert "--knowledge" not in calls[0][0]
    assert calls[1][0][calls[1][0].index("--knowledge") + 1] == str(probe.CONTROLS)
    assert probe.CONTROLS.is_file()
    assert all(call[1]["check"] for call in calls)
    summary = json.loads((output / "summary.json").read_text())
    assert [item["variant"] for item in summary["variants"]] == list(probe.VARIANTS)
    assert all(item["reached"] == item["recognized"] == 1 and
               item["median_turn_ms"] == 100 for item in summary["variants"])


def test_local_server_closes_and_stops_after_model_error(tmp_path, monkeypatch):
    weights, projector = tmp_path / "weights.gguf", tmp_path / "projector.gguf"
    weights.touch()
    projector.touch()
    monkeypatch.setitem(probe.MODELS, "jev-0.8b", (probe.MODELS["jev-0.8b"][0],
                                                  (weights, projector)))
    servers = []

    class Server:
        def __init__(self, gguf, mmproj, output):
            assert (gguf, mmproj) == (str(weights), str(projector))
            self.closed = False
            servers.append(self)

        def start(self):
            return "http://127.0.0.1:1234"

        def close(self):
            self.closed = True

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["QEV_BASE_URL"] == "http://127.0.0.1:1234"
        assert command.count("--model-file") == 2
        write_result(command, "model_error")

    monkeypatch.setattr(probe, "ModelServer", Server)
    monkeypatch.setattr(probe.subprocess, "run", run)
    output = tmp_path / "probe"
    with pytest.raises(SystemExit, match="model errors"):
        probe.main(["--model", "jev-0.8b", "--case", "square_down",
                    "--output", str(output)])
    assert len(calls) == 1
    assert servers[0].closed
    assert json.loads((output / "summary.json").read_text())["variants"][0]["model_errors"] == 1
