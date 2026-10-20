"""The model the agent asks for each action, and its chat-completions implementation."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.request import Request, urlopen

from emulator import Frame
from mystery_dungeon import CONFIG
from mystery_dungeon.actions import ACTIONS, Action, name
from mystery_dungeon.prompts import ACTION_DESCRIPTIONS, INSTRUCTIONS

if TYPE_CHECKING:
    from mystery_dungeon.agent import Context

TIMEOUT_SECONDS = 120
# Reasoning comes first so the model writes it before it commits to an action.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"},
                   "action": {"type": "string", "enum": list(ACTIONS)}},
    "required": ["reasoning", "action"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Reply:
    """The model's action and its reasoning."""

    action: Action
    reasoning: str


class Model(Protocol):
    """Choose the next action from the context and current frame."""

    name: str
    last_response: dict | None  # raw response to the latest call, for the run log

    def call(self, context: Context, frame: Frame) -> Reply:
        """Return the next action, or Done when the task is complete.

        Raises:
            Exception: when no valid action comes back.
        """
        ...


@dataclass(frozen=True)
class ChatSettings:
    """The [model] table of config.toml."""

    url: str
    name: str
    api_key_env: str = ""
    max_tokens: int = 4096
    request: dict = field(default_factory=dict)  # extra fields sent with every request


def read_settings(path: Path = CONFIG) -> ChatSettings:
    """Read the [model] table of a TOML config."""
    return ChatSettings(**tomllib.loads(path.read_text())["model"])


class ChatModel:
    """A model behind an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, settings: ChatSettings) -> None:
        self._settings = settings
        self._api_key = os.getenv(settings.api_key_env) if settings.api_key_env else None
        if settings.api_key_env and not self._api_key:
            raise ValueError(f"{settings.api_key_env} is required for {settings.name}")
        self.name = settings.name
        self.last_response: dict | None = None

    def call(self, context: Context, frame: Frame) -> Reply:
        answer = self._send(_content(context, frame))
        if not isinstance(answer, dict) or answer.keys() != ANSWER_SCHEMA["properties"].keys():
            raise ValueError(f"{self.name} returned an invalid answer: {answer!r}")
        if not isinstance(answer["action"], str) or answer["action"] not in ACTIONS:
            raise ValueError(f"{self.name} returned an unknown action: {answer['action']!r}")
        if not isinstance(answer["reasoning"], str) or not answer["reasoning"].strip():
            raise ValueError(f"{self.name} returned no reasoning: {answer['reasoning']!r}")
        return Reply(ACTIONS[answer["action"]], answer["reasoning"].strip())

    def _send(self, content: list[dict]) -> object:
        """Post one request and return the model's answer as decoded JSON."""
        self.last_response = None
        payload = {
            **self._settings.request,
            "model": self._settings.name,
            "messages": [{"role": "system", "content": INSTRUCTIONS},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "answer", "strict": True, "schema": ANSWER_SCHEMA}},
            "max_tokens": self._settings.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(self._settings.url, data=json.dumps(payload).encode(), headers=headers)
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = json.load(response)
        self.last_response = data
        if data["choices"][0].get("finish_reason") == "length":
            raise ValueError(f"{self.name} exhausted max_tokens before answering")
        try:
            return json.loads(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{self.name} returned an invalid answer") from exc


def _content(context: Context, frame: Frame) -> list[dict]:
    """Build the user message: task, knowledge and actions, earlier turns, current frame."""
    text = [f"Task: {context.task}"]
    if context.knowledge:
        text.append(f"Knowledge: {context.knowledge}")
    text.append("Actions:\n" + "\n".join(
        f"{action}: {description}" for action, description in ACTION_DESCRIPTIONS.items()))
    content = [{"type": "text", "text": "\n\n".join(text)}]
    for index, entry in enumerate(context.memory, 1):
        if entry.reasoning is None:
            after = f"Operator action, not yours: {name(entry.action)}"
        else:
            after = f"Your reasoning: {entry.reasoning}\nYour action: {name(entry.action)}"
        content += [{"type": "text", "text": f"Earlier screenshot {index}:"},
                    _image(entry.observation.frame), {"type": "text", "text": after}]
    return content + [{"type": "text", "text": "Current screenshot:"}, _image(frame)]


def _image(frame: Frame) -> dict:
    return {"type": "image_url", "image_url": {"url": frame.data_url}}
