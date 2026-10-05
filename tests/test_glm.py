"""GLM adapter sends ordered screenshots and reads a structured decision."""

import io
import json

from emulator import Frame
from mystery_dungeon.agent import OPTIONS, RecentStep, load_decision_model
from mystery_dungeon.decision_models import glm


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
