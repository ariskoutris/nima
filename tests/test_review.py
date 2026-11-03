"""The chat reviewer reads a task's turns in chunks and returns typed findings."""

import io
import json
import threading

import pytest
from fakes import FRAME, ScriptedReviewer

from mystery_dungeon import model
from mystery_dungeon.actions import Press, Stop
from mystery_dungeon.agent import Context, Turn
from mystery_dungeon.model import ChatSettings
from mystery_dungeon.review import (
    REVIEW_INSTRUCTIONS,
    REVIEW_MAX_TOKENS,
    REVIEW_TURNS,
    ChatReviewer,
    Finding,
    ReviewError,
    review,
)
from mystery_dungeon.session import Character, GameState, Observation

SETTINGS = ChatSettings("http://endpoint/v1/chat/completions", "vendor/model", "TEST_KEY")
STATE = GameState((100, 100), None, "down", characters=(
    Character("Lombre", (100, 116), "up"),))


PRESS_A = Press("A")


def turn(reasoning="Lombre is below.", action=PRESS_A) -> Turn:
    observation = Observation(FRAME, "000001.png", STATE)
    return Turn(observation, Context("Talk to Lombre", "", (), STATE), action, reasoning, "", None)


def answering(monkeypatch, answer):
    """Answer every request with the same findings; return the requests sent."""
    sent, lock = [], threading.Lock()

    def fake_urlopen(request, timeout):
        with lock:
            sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(answer)}}]}).encode())

    monkeypatch.setenv("TEST_KEY", "secret")
    monkeypatch.setattr(model, "urlopen", fake_urlopen)
    return sent


def test_turns_are_reviewed_in_chunks_with_their_screens_states_and_reasoning(monkeypatch):
    sent = answering(monkeypatch, {"findings": [
        {"kind": "dialogue", "text": " Lombre: 'Hello\n there.' "}]})
    turns = [turn() for _ in range(REVIEW_TURNS)] + [turn("Done.", Stop("complete"))]
    findings = ChatReviewer(SETTINGS).review("Talk to Lombre", turns)
    assert findings == [Finding("dialogue", "Lombre: 'Hello there.'")] * 2  # one per chunk
    first, last = sorted(sent, key=lambda payload: len(payload["messages"][1]["content"]),
                         reverse=True)
    assert first["messages"][0]["content"] == REVIEW_INSTRUCTIONS
    assert first["max_tokens"] == REVIEW_MAX_TOKENS  # dialogue is copied in full
    assert first["response_format"]["json_schema"]["schema"]["required"] == ["findings"]
    content = last["messages"][1]["content"]
    assert [part.get("text") for part in content] == [
        "# Task\nTalk to Lombre", f"## Turn {REVIEW_TURNS + 1}", None,
        "Game state: player position (100, 100) pixels, facing down\n"
        "Characters on screen, offset from you in pixels:\n"
        "- Lombre: 16 down, facing up\n"
        "Reasoning: Done.\nAction: STOP_COMPLETE"]
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_no_turns_need_no_request(monkeypatch):
    sent = answering(monkeypatch, {"findings": []})
    assert ChatReviewer(SETTINGS).review("Talk to Lombre", []) == []
    assert sent == []


@pytest.mark.parametrize("answer", [
    {"findings": [{"kind": "rumour", "text": "Lombre likes rain."}]},
    {"findings": [{"kind": "dialogue", "text": " "}]},
    {"findings": [{"kind": "dialogue"}]},
    {"findings": "Lombre: 'Hello.'"},
    {"facts": []},
])
def test_invalid_reviews_raise(monkeypatch, answer):
    answering(monkeypatch, answer)
    with pytest.raises(ValueError, match="invalid"):
        ChatReviewer(SETTINGS).review("Talk to Lombre", [turn()])


def test_a_failed_review_becomes_a_review_error():
    reviewer = ScriptedReviewer(error=TimeoutError("timed out"))
    assert review(reviewer, "Talk to Lombre", [turn()]) == ReviewError("TimeoutError: timed out")
    assert reviewer.reviews == [("Talk to Lombre", (turn(),))]
