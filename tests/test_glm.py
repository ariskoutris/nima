"""GLM adapter sends ordered screenshots and reads a structured decision."""

import io
import json

from emulator import Frame
from mystery_dungeon.agent import Agent, OPTIONS, RecentStep, load_decision_model
from mystery_dungeon.decision_models import glm
from mystery_dungeon.session import GameplaySession


def test_glm_request_and_decision(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps({"choices": [{"message": {
            "content": json.dumps({"button": "DOWN", "completion": "incomplete"})
        }}]}).encode())

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(glm, "urlopen", fake_urlopen)
    model = load_decision_model("mystery_dungeon.decision_models.glm:GlmDecisionModel")
    frame = Frame(4, 4, bytes(4 * 4 * 4))
    result = model.score("Find the exit", "Avoid walls", frame,
                         (RecentStep(frame, "UP", "before.png"),), OPTIONS)

    request = captured["request"]
    payload = json.loads(request.data)
    parts = payload["messages"][1]["content"]
    assert payload["model"] == "z-ai/glm-5.3-flash"
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["button"]["enum"] == list(OPTIONS)
    assert payload["provider"]["require_parameters"] is True
    assert payload["reasoning"]["effort"] == "low"
    assert "Find the exit" in parts[0]["text"] and "Avoid walls" in parts[0]["text"]
    assert "then pressed UP" in parts[1]["text"]
    assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[4]["image_url"]["url"].startswith("data:image/png;base64,")
    assert request.get_header("Authorization") == "Bearer test-key"
    assert result.button_scores["DOWN"] == 1.0
    assert result.completion_scores["incomplete"] == 1.0


def test_glm_journals_reasoning_and_token_limit_response(tmp_path, monkeypatch):
    responses = iter((
        {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"button": "A", "completion": "incomplete"}),
            "reasoning": "The NPC is in front of the player."}}],
         "usage": {"completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 40}}},
        {"choices": [{"finish_reason": "length", "message": {
            "content": "", "reasoning": "Still thinking about the NPC."}}],
         "usage": {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4096}}},
    ))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(glm, "urlopen",
                        lambda request, timeout: io.BytesIO(json.dumps(next(responses)).encode()))

    frame = Frame(4, 4, bytes(4 * 4 * 4))

    class Game:
        def screenshot(self):
            return frame

        def step(self, frames=1, buttons=()):
            pass

    model = glm.GlmDecisionModel()
    agent = Agent(model, tmp_path, "mystery_dungeon.decision_models.glm:GlmDecisionModel")
    agent.set_goal("Talk to the NPC")
    session = GameplaySession(Game(), tmp_path)
    session.capture()
    assert agent.step(session).status == "acted"
    assert agent.step(session).status == "model_error"

    journal = (tmp_path / "decisions.jsonl").read_text()
    rows = [json.loads(line) for line in journal.splitlines()]
    action = next(row for row in rows if row["event"] == "action")
    error = next(row for row in rows if row["event"] == "model_error")
    assert action["model_response"]["choices"][0]["message"]["reasoning"].startswith("The NPC")
    assert action["model_response"]["usage"]["completion_tokens"] == 50
    assert error["model_response"]["choices"][0]["finish_reason"] == "length"
    assert error["model_response"]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4096
    assert "max_tokens" in error["error"]
    assert "test-key" not in journal
