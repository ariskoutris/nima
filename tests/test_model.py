"""The chat model sends the instructions with ordered screenshots and reads typed answers."""

import io
import json

import pytest
from fakes import FRAME

from emulator import Frame
from mystery_dungeon import model
from mystery_dungeon.actions import Done, Press
from mystery_dungeon.agent import Context, MemoryEntry
from mystery_dungeon.model import (
    ChatModel,
    ChatSettings,
    Reply,
    internal_reasoning,
    read_settings,
    scaled,
)
from mystery_dungeon.prompts import INSTRUCTIONS
from mystery_dungeon.session import Observation

MEMORY = (MemoryEntry(Observation(FRAME, "before.png"), Press("UP"), "Walk up to the door."),
          MemoryEntry(Observation(FRAME, "manual.png"), Press("A"), None))
SETTINGS = ChatSettings("http://endpoint/v1/chat/completions", "vendor/model", "TEST_KEY", 512,
                        {"reasoning": {"effort": "low"}})


def answering(monkeypatch, *responses):
    """Answer requests in order; return the requests sent."""
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


def test_request_carries_instructions_settings_and_memory(monkeypatch):
    sent = answering(monkeypatch, reply({"reasoning": " The door is ahead. ", "action": "DOWN"}))
    chat_model = ChatModel(SETTINGS)
    answer = chat_model.call(Context("Find the exit", "Avoid walls", MEMORY), FRAME)
    assert answer == Reply(Press("DOWN"), "The door is ahead.")
    assert sent[0].full_url == "http://endpoint/v1/chat/completions"
    assert sent[0].get_header("Authorization") == "Bearer secret"
    payload = json.loads(sent[0].data)
    assert payload["model"] == "vendor/model" and payload["max_tokens"] == 512
    assert payload["reasoning"]["effort"] == "low"  # the settings' request extras
    system, user = payload["messages"]
    assert system["content"] == INSTRUCTIONS
    assert "# Task\nFind the exit" in user["content"][0]["text"]
    assert "# Knowledge\nAvoid walls" in user["content"][0]["text"]
    assert user["content"][0]["text"].endswith("# Earlier turns (oldest first)")
    texts = [part.get("text") for part in user["content"]]
    assert texts == [user["content"][0]["text"],
                     "## Turn -2", None, "Reasoning: Walk up to the door.\nAction: UP",
                     "## Turn -1 (operator)", None, "Action: A",
                     "# Current screenshot", None]
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,")
               for part in user["content"] if part["type"] == "image_url")
    schema = payload["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["reasoning", "action"]
    assert schema["required"] == ["reasoning", "action"]
    assert {"WAIT", "DONE"} <= set(schema["properties"]["action"]["enum"])
    assert chat_model.last_response["choices"][0]["message"]["reasoning"] == "The NPC is ahead."


def test_done_reports_completion_with_its_reasoning(monkeypatch):
    answering(monkeypatch, reply({"reasoning": "The dialogue closed.", "action": "DONE"}))
    answer = ChatModel(SETTINGS).call(Context("Find the exit", "", MEMORY), FRAME)
    assert answer == Reply(Done(), "The dialogue closed.")


def test_endpoints_without_a_key_get_no_authorization(monkeypatch):
    sent = answering(monkeypatch, reply({"reasoning": "Wait.", "action": "WAIT"}))
    local = ChatModel(ChatSettings("http://localhost/v1/chat/completions", "local"))
    local.call(Context("Talk", "", ()), FRAME)
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


@pytest.mark.parametrize("response,message", [
    (reply({"reasoning": "Start.", "action": "START"}), "unknown action"),
    (reply({"reasoning": "Press A.", "action": ["A"]}), "unknown action"),
    (reply({"reasoning": " ", "action": "A"}), "no reasoning"),
    (reply({"action": "A"}), "invalid answer"),
    (reply({"button": "A"}), "invalid answer"),
    (reply({}, finish_reason="length"), "max_tokens"),
])
def test_invalid_answers_raise_and_keep_the_response(monkeypatch, response, message):
    answering(monkeypatch, response)
    chat_model = ChatModel(SETTINGS)
    with pytest.raises(ValueError, match=message):
        chat_model.call(Context("Talk", "", ()), FRAME)
    assert chat_model.last_response == response


@pytest.mark.parametrize("message, expected", [
    ({"content": "{}", "reasoning": "OpenRouter."}, "OpenRouter."),
    ({"content": "{}", "reasoning_content": "vLLM."}, "vLLM."),
    ({"content": "{}"}, None),
])
def test_internal_reasoning_reads_either_provider_field(message, expected):
    assert internal_reasoning({"choices": [{"message": message}]}) == expected
    assert internal_reasoning(None) is None


def test_scaled_repeats_each_pixel_across_and_down():
    red, blue = bytes([255, 0, 0, 255]), bytes([0, 0, 255, 255])
    big = scaled(Frame(2, 1, red + blue), 2)
    assert (big.width, big.height) == (4, 2)
    assert big.rgba == (red * 2 + blue * 2) * 2
