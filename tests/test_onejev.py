"""OneJev asks shared prompts over ordered frames and returns typed answers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fakes import FRAME

from mystery_dungeon.actions import Press
from mystery_dungeon.agent import MemoryEntry
from mystery_dungeon.models import load_model, onejev
from mystery_dungeon.prompts import ACT, COMPLETE

MEMORY = (MemoryEntry(FRAME, "first.png", Press("DOWN")),)
URL = "http://127.0.0.1:9000"


def client_answering(monkeypatch, answer):
    client = Mock()
    client.system_one.return_value = SimpleNamespace(
        answers={"answer": SimpleNamespace(raw={"type": "test"}, **answer)}, usage={"tokens": 3})
    monkeypatch.setattr(onejev, "Client", lambda url: client)
    return client


def test_action_question_carries_task_memory_and_knowledge(monkeypatch):
    client = client_answering(monkeypatch, {"choice": "A"})
    model = load_model("mystery_dungeon.models.onejev:OneJevModel", URL)
    assert model.choose_action("Find exit", "Use the D-pad", FRAME, MEMORY) == Press("A")
    kwargs = client.system_one.call_args.kwargs
    assert len(kwargs["media"]) == 2
    assert kwargs["state"]["task"] == "Find exit"
    assert kwargs["state"]["observations"][0]["action_after"] == "DOWN"
    assert kwargs["questions"]["answer"].instructions == f"{ACT}\nUse the D-pad"
    assert "WAIT" in kwargs["questions"]["answer"].criteria
    assert model.last_response == {"answer": {"type": "test"}, "usage": {"tokens": 3}}


@pytest.mark.parametrize("p_yes,complete", [(0.8, True), (0.5, False)])
def test_completion_is_a_yes_no_question(monkeypatch, p_yes, complete):
    client = client_answering(monkeypatch, {"noul": p_yes})
    assert onejev.OneJevModel(URL).is_complete("Find exit", FRAME, MEMORY) is complete
    question = client.system_one.call_args.kwargs["questions"]["answer"]
    assert question.instructions == COMPLETE


def test_unknown_action_raises(monkeypatch):
    client_answering(monkeypatch, {"choice": "START"})
    with pytest.raises(ValueError, match="unknown action"):
        onejev.OneJevModel(URL).choose_action("Find exit", "", FRAME, ())


def test_missing_server_url_raises():
    with pytest.raises(ValueError, match="needs a server"):
        onejev.OneJevModel()
