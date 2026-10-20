"""The chat model sends the instructions with ordered screenshots and reads typed answers."""

import io
import json

import pytest
from fakes import FRAME

from emulator import Frame
from mystery_dungeon import model
from mystery_dungeon.actions import Press, Stop
from mystery_dungeon.agent import Context, MemoryEntry
from mystery_dungeon.model import (
    ChatModel,
    ChatSettings,
    Finding,
    Replacement,
    Reply,
    ScratchpadEdit,
    internal_reasoning,
    read_settings,
    scaled,
    token_usage,
)
from mystery_dungeon.prompts import INSTRUCTIONS
from mystery_dungeon.session import Character, GameState, Observation

MEMORY = (MemoryEntry(Observation(FRAME, "before.png"), Press("UP"), "Walk up to the door."),
          MemoryEntry(Observation(FRAME, "manual.png"), Press("A"), None))
SETTINGS = ChatSettings("http://endpoint/v1/chat/completions", "vendor/model", "TEST_KEY", 512,
                        {"reasoning": {"effort": "low"}})
EMPTY = {"scratchpad": {"add": [], "remove": [], "replace": []}, "findings": []}
CHARIZARD = {"kind": "dialogue",
             "text": " Charizard in Pokémon Square says its team reached Gold Rank. "}


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
    sent = answering(monkeypatch, reply({
        "reasoning": " The door is ahead. ",
        "scratchpad": {"add": [" Found the door. "], "remove": [2],
                       "replace": [{"line": 1, "text": " Left the house, a guess. "}]},
        "findings": [CHARIZARD], "action": "DOWN"}))
    chat_model = ChatModel(SETTINGS)
    context = Context("Find the exit", "Avoid walls", MEMORY,
                      scratchpad=("Left the house.", "The exit is south, a guess."))
    answer = chat_model.call(context, FRAME)
    assert answer == Reply(Press("DOWN"), "The door is ahead.",
                           ScratchpadEdit(("Found the door.",), (2,),
                                          (Replacement(1, "Left the house, a guess."),)),
                           (Finding("dialogue", CHARIZARD["text"].strip()),))
    assert sent[0].full_url == "http://endpoint/v1/chat/completions"
    assert sent[0].get_header("Authorization") == "Bearer secret"
    payload = json.loads(sent[0].data)
    assert payload["model"] == "vendor/model" and payload["max_tokens"] == 512
    assert payload["reasoning"]["effort"] == "low"  # the settings' request extras
    system, user = payload["messages"]
    assert system["content"] == INSTRUCTIONS
    assert "# Task\nFind the exit" in user["content"][0]["text"]
    assert "# Knowledge\nAvoid walls" in user["content"][0]["text"]
    assert ("# Your scratchpad\n[1] Left the house.\n[2] The exit is south, a guess.\n\n"
            "# Actions" in user["content"][0]["text"])
    assert user["content"][0]["text"].endswith("# Earlier turns (oldest first)")
    texts = [part.get("text") for part in user["content"]]
    assert texts == [user["content"][0]["text"],
                     "## Turn -2", None, "Reasoning: Walk up to the door.\nAction: UP",
                     "## Turn -1 (operator)", None, "Action: A",
                     "# Current screenshot", None]
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,")
               for part in user["content"] if part["type"] == "image_url")
    schema = payload["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["reasoning", "scratchpad", "findings", "action"]
    assert schema["required"] == ["reasoning", "scratchpad", "findings", "action"]
    assert schema["properties"]["findings"]["items"]["properties"]["kind"]["enum"] == [
        "dialogue", "location", "mechanic", "other"]
    assert {"WAIT", "STOP_STUCK"} <= set(schema["properties"]["action"]["enum"])
    assert chat_model.last_response["choices"][0]["message"]["reasoning"] == "The NPC is ahead."


@pytest.mark.parametrize("action,outcome", [
    ("STOP_COMPLETE", "complete"), ("STOP_STUCK", "stuck"), ("STOP_INVALID", "invalid")])
def test_each_stop_action_carries_its_outcome(monkeypatch, action, outcome):
    answering(monkeypatch, reply({"reasoning": "The dialogue closed.", **EMPTY,
                                  "action": action}))
    answer = ChatModel(SETTINGS).call(Context("Find the exit", "", MEMORY), FRAME)
    assert answer == Reply(Stop(outcome), "The dialogue closed.", ScratchpadEdit(), ())


def test_an_empty_scratchpad_is_shown_as_empty(monkeypatch):
    sent = answering(monkeypatch, reply({"reasoning": "Wait.", **EMPTY, "action": "WAIT"}))
    assert ChatModel(SETTINGS).call(Context("Talk", "", ()), FRAME).findings == ()
    text = json.loads(sent[0].data)["messages"][1]["content"][0]["text"]
    assert "# Your scratchpad\n(empty)\n\n# Actions" in text


def test_endpoints_without_a_key_get_no_authorization(monkeypatch):
    sent = answering(monkeypatch, reply({"reasoning": "Wait.", **EMPTY, "action": "WAIT"}))
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
    (reply({"reasoning": "Start.", **EMPTY, "action": "START"}), "unknown action"),
    (reply({"reasoning": "Press A.", **EMPTY, "action": ["A"]}), "unknown action"),
    (reply({"reasoning": " ", **EMPTY, "action": "A"}), "no reasoning"),
    (reply({"reasoning": "Press A.", **EMPTY, "findings": None, "action": "A"}),
     "invalid findings"),
    (reply({"reasoning": "Press A.", **EMPTY, "findings": [{**CHARIZARD, "kind": "item"}],
            "action": "A"}), "invalid finding"),
    (reply({"reasoning": "Press A.", **EMPTY, "findings": [{**CHARIZARD, "text": " "}],
            "action": "A"}), "invalid finding"),
    (reply({"reasoning": "Press A.", **EMPTY, "findings": ["Charizard is Gold Rank."],
            "action": "A"}), "invalid finding"),
    *[(reply({"reasoning": "Press A.", **EMPTY, "action": "A",
                "scratchpad": {"add": [], "remove": [], "replace": [], **edit}}),
       "invalid scratchpad edit") for edit in (
        {"replace": None}, {"add": [""]}, {"remove": ["1"]}, {"remove": [True]},
        {"remove": [2, 2]}, {"remove": [2], "replace": [{"line": 2, "text": "Twice."}]},
        {"replace": [{"line": 1}]}, {"add": ["[4] Talked to Kecleon."]},
        {"replace": [{"line": 1, "text": " [1] Talked to Kecleon."}]})],
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


def test_token_usage_splits_reasoning_out_of_output():
    usage = {"prompt_tokens": 900, "completion_tokens": 60,
             "completion_tokens_details": {"reasoning_tokens": 20}}
    assert token_usage({"usage": usage}) == {"prompt": 900, "output": 40, "reasoning": 20}


def test_token_usage_without_reasoning_or_usage():
    usage = {"prompt_tokens": 900, "completion_tokens": 60}
    assert token_usage({"usage": usage}) == {"prompt": 900, "output": 60, "reasoning": None}
    assert token_usage({"choices": []}) is None
    assert token_usage(None) is None


def test_scaled_repeats_each_pixel_across_and_down():
    red, blue = bytes([255, 0, 0, 255]), bytes([0, 0, 255, 255])
    big = scaled(Frame(2, 1, red + blue), 2)
    assert (big.width, big.height) == (4, 2)
    assert big.rgba == (red * 2 + blue * 2) * 2


def test_game_state_follows_each_screenshot_it_belongs_to(monkeypatch):
    sent = answering(monkeypatch, reply({"reasoning": "Blocked.", **EMPTY, "action": "DOWN"}))
    memory = (MemoryEntry(Observation(FRAME, "before.png", GameState((120, 88), None, "west")),
                          Press("LEFT"), "Step left."),)
    context = Context("Find the exit", "", memory, GameState(None, (4, 7), "west"))
    ChatModel(SETTINGS).call(context, FRAME)
    user = json.loads(sent[0].data)["messages"][1]["content"]
    texts = [part.get("text") for part in user[1:]]
    assert texts == ["## Turn -1", None,
                     "Game state: player position (120, 88) pixels, facing west\n"
                     "Reasoning: Step left.\nAction: LEFT",
                     "# Current screenshot", None,
                     "# Game state\nplayer tile (4, 7), facing west"]


def test_on_screen_characters_are_described_relative_to_the_player(monkeypatch):
    sent = answering(monkeypatch, reply({"reasoning": "Face it.", **EMPTY, "action": "DOWN"}))
    state = GameState((464, 332), None, "west", (
        Character("Granbull", (456, 352), "south"), Character("Lombre", (464, 332), "east")),
        "Quilava")
    ChatModel(SETTINGS).call(Context("Talk", "", (), state), FRAME)
    user = json.loads(sent[0].data)["messages"][1]["content"]
    assert user[-1]["text"] == (
        "# Game state\nYou play as Quilava.\nplayer position (464, 332) pixels, facing west\n"
        "Characters on screen, relative to you:\n"
        "- Granbull: 8 left, 20 down, facing south\n"
        "- Lombre: same position, facing east")
