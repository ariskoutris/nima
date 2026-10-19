"""d1 asks shared prompts over ordered screenshots and reads typed answers."""

import io
import json
from urllib.error import HTTPError

import pytest
from fakes import FRAME

from mystery_dungeon.actions import Press
from mystery_dungeon.agent import MemoryEntry
from mystery_dungeon.models import d1, load_model
from mystery_dungeon.prompts import ACT, COMPLETE

MEMORY = (MemoryEntry(FRAME, "before.png", Press("UP")),)
URL = "http://127.0.0.1:8080"


def answering(monkeypatch, response):
    sent = []

    def fake_urlopen(request, timeout):
        sent.append((request.full_url, json.loads(request.data)))
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(d1, "urlopen", fake_urlopen)
    return sent


def reply(answer):
    return {"answers": {"answer": answer}}


def test_action_request_carries_task_memory_and_knowledge(monkeypatch):
    response = reply({"type": "choice", "choice": "DOWN"})
    sent = answering(monkeypatch, response)
    model = load_model("mystery_dungeon.models.d1:D1Model", URL)
    assert model.choose_action("Find the exit", "Avoid walls", FRAME, MEMORY) == Press("DOWN")
    url, payload = sent[0]
    assert url == "http://127.0.0.1:8080/v1/systemone"
    assert len(payload["images"]) == 2
    state = json.loads(payload["state"])
    assert state["task"] == "Find the exit"
    assert state["observations"] == [{"image": 1, "action_after": "UP"},
                                     {"image": 2, "role": "current"}]
    question = payload["questions"]["answer"]
    assert question["type"] == "choice"
    assert question["instructions"] == f"{ACT}\nAvoid walls"
    assert "WAIT" in question["criteria"]
    assert model.last_response == response


@pytest.mark.parametrize("p_yes,complete", [(0.8, True), (0.5, False)])
def test_completion_is_a_yes_no_question(monkeypatch, p_yes, complete):
    sent = answering(monkeypatch, reply({"type": "noul", "noul": p_yes}))
    assert d1.D1Model(URL).is_complete("Find the exit", FRAME, MEMORY) is complete
    assert sent[0][1]["questions"]["answer"] == {"type": "noul", "instructions": COMPLETE}


def test_server_url_is_passed_in(monkeypatch):
    sent = answering(monkeypatch, reply({"type": "noul", "noul": 0.1}))
    d1.D1Model("http://127.0.0.1:8090/").is_complete("Talk", FRAME, ())
    assert sent[0][0] == "http://127.0.0.1:8090/v1/systemone"


@pytest.mark.parametrize("response,error", [
    (reply({"type": "choice", "choice": "START"}), ValueError),
    ({"error": "model not loaded"}, KeyError),
])
def test_invalid_action_answers_raise_and_keep_the_response(monkeypatch, response, error):
    answering(monkeypatch, response)
    model = d1.D1Model(URL)
    with pytest.raises(error):
        model.choose_action("Talk", "", FRAME, ())
    assert model.last_response == response


def test_completion_without_a_probability_raises(monkeypatch):
    answering(monkeypatch, reply({"type": "choice", "choice": "A"}))
    with pytest.raises(KeyError):
        d1.D1Model(URL).is_complete("Talk", FRAME, ())


def test_server_errors_carry_the_response_body(monkeypatch):
    def refuse(request, timeout):
        raise HTTPError(request.full_url, 503, "Service Unavailable", {},
                        io.BytesIO(b'{"error": "model not loaded"}'))

    monkeypatch.setattr(d1, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="503.*model not loaded"):
        d1.D1Model(URL).is_complete("Talk", FRAME, ())


def test_missing_server_url_raises():
    with pytest.raises(ValueError, match="needs a server"):
        d1.D1Model()
