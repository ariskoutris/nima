"""GLM-5.3-Flash vision decisions through OpenRouter."""

from __future__ import annotations

import base64
import json
import os
from typing import Mapping
from urllib.request import Request, urlopen

from emulator import Frame
from mystery_dungeon.agent import COMPLETION, DecisionScores, RecentStep


MODEL = "z-ai/glm-5.3-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = (
    "You control a Pokémon Mystery Dungeon game using screenshots. Choose exactly one "
    "next button to make progress toward the goal. Earlier screenshots are in time order; "
    "each named action happened after its screenshot. Judge goal completion from the "
    "visible evidence only. Use 'complete' only when the goal is clearly achieved, "
    "'incomplete' when it is clearly not achieved, and 'uncertain' otherwise. "
    "Don't overthink it. Just return the requested JSON object."
)


class GlmDecisionModel:
    prompt_version = "v1"
    prompt_bytes = json.dumps({"version": prompt_version, "system": SYSTEM_PROMPT}).encode()

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for GLM-5.3-Flash")
        self.last_response: dict | None = None

    def score(self, goal: str, guidance: str, current_frame: Frame,
              recent_steps: tuple[RecentStep, ...],
              button_options: Mapping[str, str]) -> DecisionScores:
        self.last_response = None
        instructions = [f"Goal: {goal}"]
        if guidance:
            instructions.append(f"Guidance: {guidance}")
        instructions.append("Available buttons:\n" + "\n".join(
            f"{button}: {description}" for button, description in button_options.items()))
        content = [{"type": "text", "text": "\n\n".join(instructions)}]
        for index, step in enumerate(recent_steps, 1):
            content.extend((
                {"type": "text", "text": f"Earlier screenshot {index}; then pressed {step.action}:"},
                self._image(step.frame),
            ))
        content.extend((
            {"type": "text", "text": "Current screenshot; choose the next button:"},
            self._image(current_frame),
        ))
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "game_decision", "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "button": {"type": "string", "enum": list(button_options)},
                            "completion": {"type": "string", "enum": list(COMPLETION)},
                        },
                        "required": ["button", "completion"],
                        "additionalProperties": False,
                    },
                },
            },
            "provider": {"require_parameters": True},
            "reasoning": {"effort": "low"},
            "max_tokens": 4096,
        }
        request = Request(URL, data=json.dumps(payload).encode(), headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        with urlopen(request, timeout=120) as response:
            data = json.load(response)
        self.last_response = data
        if data["choices"][0].get("finish_reason") == "length":
            raise ValueError("GLM exhausted max_tokens before finishing its decision")
        try:
            answer = json.loads(data["choices"][0]["message"]["content"])
            button, completion = answer["button"], answer["completion"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("GLM returned an invalid decision") from exc
        if (not isinstance(button, str) or button not in button_options or
                not isinstance(completion, str) or completion not in COMPLETION):
            raise ValueError(f"GLM returned an unknown button or completion: {answer}")
        return DecisionScores(
            {name: float(name == button) for name in button_options},
            {name: float(name == completion) for name in COMPLETION},
        )

    @staticmethod
    def _image(frame: Frame) -> dict:
        encoded = base64.b64encode(frame.png_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
