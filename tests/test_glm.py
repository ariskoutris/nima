"""GLM sends shared prompts with ordered screenshots and reads typed answers."""

import io
import json

import pytest
from fakes import FRAME

from mystery_dungeon.actions import Press
from mystery_dungeon.agent import MemoryEntry
from mystery_dungeon.models import glm
from mystery_dungeon.prompts import ACT, COMPLETE

MEMORY = (MemoryEntry(FRAME, "before.png", Press("UP")),)


def answering(monkeypatch, *responses):
    sent = []
    replies = iter(responses)

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps(next(replies)).encode())

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(glm, "urlopen", fake_urlopen)
    return sent


def reply(answer, finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason, "message": {
        "content": json.dumps(answer), "reasoning": "The NPC is ahead."}}],
        "usage": {"completion_tokens": 50}}


def test_action_request_uses_shared_prompt_knowledge_and_memory(monkeypatch):
    sent = answering(monkeypatch, reply({"action": "DOWN"}))
    model = glm.GlmModel()
    assert model.choose_action("Find the exit", "Avoid walls", FRAME, MEMORY) == Press("DOWN")
    payload = sent[0]
    system, user = payload["messages"]
    assert system["content"] == ACT
    assert "Task: Find the exit" in user["content"][0]["text"]
    assert "Knowledge: Avoid walls" in user["content"][0]["text"]
    assert "followed by UP" in user["content"][1]["text"]
    assert user["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user["content"][4]["image_url"]["url"].startswith("data:image/png;base64,")
    schema = payload["response_format"]["json_schema"]["schema"]["properties"]["action"]
    assert "WAIT" in schema["enum"]
    assert payload["reasoning"]["effort"] == "low"
    assert model.last_response["choices"][0]["message"]["reasoning"] == "The NPC is ahead."


def test_completion_request_is_boolean_and_has_no_knowledge(monkeypatch):
    sent = answering(monkeypatch, reply({"complete": True}))
    assert glm.GlmModel().is_complete("Find the exit", FRAME, MEMORY) is True
    system, user = sent[0]["messages"]
    assert system["content"] == COMPLETE
    assert user["content"][0]["text"] == "Task: Find the exit"
    schema = sent[0]["response_format"]["json_schema"]["schema"]["properties"]["complete"]
    assert schema == {"type": "boolean"}


@pytest.mark.parametrize("response,message", [
    (reply({"action": "START"}), "unknown action"),
    (reply({"button": "A"}), "invalid answer"),
    (reply({}, finish_reason="length"), "max_tokens"),
])
def test_invalid_answers_raise_and_keep_the_response(monkeypatch, response, message):
    answering(monkeypatch, response)
    model = glm.GlmModel()
    with pytest.raises(ValueError, match=message):
        model.choose_action("Talk", "", FRAME, ())
    assert model.last_response == response


def test_non_boolean_completion_raises(monkeypatch):
    answering(monkeypatch, reply({"complete": "yes"}))
    with pytest.raises(ValueError, match="non-boolean"):
        glm.GlmModel().is_complete("Talk", FRAME, ())
