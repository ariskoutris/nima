"""GLM-5.3-Flash through OpenRouter, filling both model roles."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

from emulator import Frame
from mystery_dungeon.actions import ACTIONS, Action, name
from mystery_dungeon.prompts import ACT, ACTION_DESCRIPTIONS, COMPLETE

if TYPE_CHECKING:
    from mystery_dungeon.agent import MemoryEntry

MODEL = "z-ai/glm-5.3-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"


class GlmModel:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for GLM-5.3-Flash")
        self.last_response: dict | None = None

    def choose_action(self, task: str, knowledge: str, frame: Frame,
                      memory: tuple[MemoryEntry, ...]) -> Action:
        text = [f"Task: {task}"]
        if knowledge:
            text.append(f"Knowledge: {knowledge}")
        text.append("Available actions:\n" + "\n".join(
            f"{action}: {description}" for action, description in ACTION_DESCRIPTIONS.items()))
        answer = self._ask(ACT, "\n\n".join(text), frame, memory,
                           "action", {"type": "string", "enum": list(ACTIONS)})
        if answer not in ACTIONS:
            raise ValueError(f"GLM returned an unknown action: {answer!r}")
        return ACTIONS[answer]

    def is_complete(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...]) -> bool:
        answer = self._ask(COMPLETE, f"Task: {task}", frame, memory,
                           "complete", {"type": "boolean"})
        if not isinstance(answer, bool):
            raise ValueError(f"GLM returned a non-boolean completion: {answer!r}")
        return answer

    def _ask(self, system: str, text: str, frame: Frame, memory: tuple[MemoryEntry, ...],
             field: str, schema: dict) -> object:
        self.last_response = None
        content = [{"type": "text", "text": text}]
        for index, entry in enumerate(memory, 1):
            content += [{"type": "text", "text": f"Earlier screenshot {index}, followed by {name(entry.action)}:"},
                        _image(entry.frame)]
        content += [{"type": "text", "text": "Current screenshot:"}, _image(frame)]
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": field, "strict": True,
                "schema": {"type": "object", "properties": {field: schema},
                           "required": [field], "additionalProperties": False},
            }},
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
            raise ValueError("GLM exhausted max_tokens before answering")
        try:
            return json.loads(data["choices"][0]["message"]["content"])[field]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("GLM returned an invalid answer") from exc


def _image(frame: Frame) -> dict:
    return {"type": "image_url", "image_url": {"url": frame.data_url}}
