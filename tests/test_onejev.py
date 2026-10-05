"""OneJev receives ordered frames and separates guidance from completion."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emulator import Frame
from mystery_dungeon.agent import RecentStep, load_decision_model
from mystery_dungeon.decision_models import onejev


def test_joint_request_uses_history_and_goal(monkeypatch):
    frame = Frame(4, 4, bytes(4 * 4 * 4))
    response = SimpleNamespace(answers={
        "action": SimpleNamespace(probabilities={"A": 1.0}),
        "completion": SimpleNamespace(probabilities={
            "complete": 0.2, "incomplete": 0.7, "uncertain": 0.1})})
    client = Mock()
    client.system_one.return_value = response
    monkeypatch.setenv("QEV_PROMPT_VERSION", "v1")
    monkeypatch.setattr(onejev, "Client", lambda: client)
    model = load_decision_model("mystery_dungeon.decision_models.onejev:OneJevDecisionModel")
    result = model.score("Find exit", "Use the D-pad", frame,
                         (RecentStep(frame, "DOWN", "first.png"),),
                         {"A": "press A"})
    kwargs = client.system_one.call_args.kwargs
    assert len(kwargs["media"]) == 2
    assert kwargs["state"]["observations"][0]["action_after"] == "DOWN"
    assert kwargs["state"]["goal"] == "Find exit"
    assert "Use the D-pad" not in str(kwargs["state"])
    assert "Use the D-pad" in kwargs["questions"]["action"].instructions
    assert "Use the D-pad" not in kwargs["questions"]["completion"].instructions
    assert kwargs["questions"]["completion"].instructions == model.prompt["completion_question"]
    assert model.prompt["version"] == "v1"
    assert result.completion_scores["incomplete"] == 0.7


def test_prompt_version_rejects_paths(monkeypatch):
    monkeypatch.setenv("QEV_PROMPT_VERSION", "../v1")
    with pytest.raises(ValueError):
        onejev.OneJevDecisionModel()
