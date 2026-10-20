"""The chat model sends shared prompts with ordered screenshots and reads typed answers."""

import io
import json

import pytest
from fakes import FRAME

from mystery_dungeon import model
from mystery_dungeon.actions import Press
from mystery_dungeon.agent import MemoryEntry
from mystery_dungeon.model import ChatModel, ChatSettings, read_settings
from mystery_dungeon.prompts import ACT, COMPLETE

MEMORY = (MemoryEntry(FRAME, "before.png", Press("UP")),)
SETTINGS = ChatSettings("http://endpoint/v1/chat/completions", "vendor/model", "TEST_KEY", 512,
                        {"reasoning": {"effort": "low"}})


def answering(monkeypatch, *responses):
    sent = []
    replies = iter(responses)

    def fake_urlopen(request, timeout):
        sent.append(request)
        return io.BytesIO(json.dumps(next(replies)).encode())

    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(model, "urlopen", fake_urlopen)
    return sent


def reply(answer, finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason, "message": {
        "content": json.dumps(answer), "reasoning": "The NPC is ahead."}}],
        "usage": {"completion_tokens": 50}}


def test_action_request_uses_shared_prompt_knowledge_and_memory(monkeypatch):
    sent = answering(monkeypatch, reply({"action": "DOWN"}))
    chat_model = ChatModel(SETTINGS)
    assert chat_model.choose_action("Find the exit", "Avoid walls", FRAME, MEMORY) == Press("DOWN")
    assert sent[0].full_url == "http://endpoint/v1/chat/completions"
    assert sent[0].get_header("Authorization") == "Bearer secret"
    payload = json.loads(sent[0].data)
    assert payload["model"] == "vendor/model" and payload["max_tokens"] == 512
    system, user = payload["messages"]
    assert system["content"] == ACT
    assert "Task: Find the exit" in user["content"][0]["text"]
    assert "Knowledge: Avoid walls" in user["content"][0]["text"]
    assert "followed by UP" in user["content"][1]["text"]
    assert user["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user["content"][4]["image_url"]["url"].startswith("data:image/png;base64,")
    schema = payload["response_format"]["json_schema"]["schema"]["properties"]["action"]
    assert "WAIT" in schema["enum"]
    assert payload["reasoning"]["effort"] == "low"  # the settings' request extras
    assert chat_model.last_response["choices"][0]["message"]["reasoning"] == "The NPC is ahead."


def test_completion_request_is_boolean_and_has_no_knowledge(monkeypatch):
    sent = answering(monkeypatch, reply({"complete": True}))
    assert ChatModel(SETTINGS).is_complete("Find the exit", FRAME, MEMORY) is True
    payload = json.loads(sent[0].data)
    system, user = payload["messages"]
    assert system["content"] == COMPLETE
    assert user["content"][0]["text"] == "Task: Find the exit"
    schema = payload["response_format"]["json_schema"]["schema"]["properties"]["complete"]
    assert schema == {"type": "boolean"}


@pytest.mark.parametrize("response,message", [
    (reply({"action": "START"}), "unknown action"),
    (reply({"button": "A"}), "invalid answer"),
    (reply({}, finish_reason="length"), "max_tokens"),
])
def test_invalid_answers_raise_and_keep_the_response(monkeypatch, response, message):
    answering(monkeypatch, response)
    chat_model = ChatModel(SETTINGS)
    with pytest.raises(ValueError, match=message):
        chat_model.choose_action("Talk", "", FRAME, ())
    assert chat_model.last_response == response


def test_non_boolean_completion_raises(monkeypatch):
    answering(monkeypatch, reply({"complete": "yes"}))
    with pytest.raises(ValueError, match="non-boolean"):
        ChatModel(SETTINGS).is_complete("Talk", FRAME, ())


def test_endpoints_without_a_key_get_no_authorization(monkeypatch):
    sent = answering(monkeypatch, reply({"action": "WAIT"}))
    local = ChatModel(ChatSettings("http://localhost/v1/chat/completions", "local"))
    local.choose_action("Talk", "", FRAME, ())
    assert sent[0].get_header("Authorization") is None
    assert local.name == "local"


def test_a_named_key_must_be_set(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_KEY is required"):
        ChatModel(SETTINGS)


def test_settings_come_from_the_model_table(tmp_path):
    settings = read_settings()  # the repository's config.toml
    assert settings.url.endswith("/chat/completions") and settings.name
    config = tmp_path / "config.toml"
    config.write_text('[model]\nurl = "http://x"\nname = "m"\ntemperature = 0\n')
    with pytest.raises(TypeError):
        read_settings(config)
