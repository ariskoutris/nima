"""Contracts for planning and action models, and the model behind a chat-completions endpoint."""

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
from mystery_dungeon.prompts import ACT, ACTION_DESCRIPTIONS, COMPLETE

if TYPE_CHECKING:
    from mystery_dungeon.agent import MemoryEntry

TIMEOUT_SECONDS = 120


class ActionModel(Protocol):
    """Chooses the next action for a task."""

    name: str  # recorded in run configs
    last_response: dict | None  # raw answer to the latest call, kept for the run log

    def choose_action(self, task: str, knowledge: str, frame: Frame,
                      memory: tuple[MemoryEntry, ...]) -> Action:
        """Return one action from ACTIONS for the current frame.

        Args:
            memory: recent screenshots, oldest first, each with the action taken after it.

        Raises:
            Exception: when the model fails or answers outside the action space.
        """
        ...


class PlanningModel(Protocol):
    """Judges progress on a task."""

    name: str
    last_response: dict | None

    def is_complete(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...]) -> bool:
        """Judge from visible evidence only whether the task has been completed.

        Raises:
            Exception: when the model fails or does not answer yes or no.
        """
        ...


class Model(ActionModel, PlanningModel, Protocol):
    """One model filling both roles."""


@dataclass(frozen=True)
class ChatSettings:
    """The [model] table: which endpoint to ask, under which name, and how."""

    url: str
    name: str
    api_key_env: str = ""
    max_tokens: int = 4096
    request: dict = field(default_factory=dict)


def read_settings(path: Path = CONFIG) -> ChatSettings:
    """Read the [model] table of a TOML config.

    Raises:
        KeyError: when the config has no [model] table.
        TypeError: when the table is missing url or name, or has an unknown key.
    """
    return ChatSettings(**tomllib.loads(path.read_text())["model"])


class ChatModel:
    """Ask a model behind an OpenAI-compatible chat-completions endpoint, in both roles."""

    def __init__(self, settings: ChatSettings) -> None:
        """Raises:
            ValueError: when the settings name an API key variable that is not set.
        """
        self._settings = settings
        self._api_key = os.getenv(settings.api_key_env) if settings.api_key_env else None
        if settings.api_key_env and not self._api_key:
            raise ValueError(f"{settings.api_key_env} is required for {settings.name}")
        self.name = settings.name
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
            raise ValueError(f"{self.name} returned an unknown action: {answer!r}")
        return ACTIONS[answer]

    def is_complete(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...]) -> bool:
        answer = self._ask(COMPLETE, f"Task: {task}", frame, memory,
                           "complete", {"type": "boolean"})
        if not isinstance(answer, bool):
            raise ValueError(f"{self.name} returned a non-boolean completion: {answer!r}")
        return answer

    def _ask(self, system: str, text: str, frame: Frame, memory: tuple[MemoryEntry, ...],
             field: str, schema: dict) -> object:
        self.last_response = None
        content = [{"type": "text", "text": text}]
        for index, entry in enumerate(memory, 1):
            label = f"Earlier screenshot {index}, followed by {name(entry.action)}:"
            content += [{"type": "text", "text": label},
                        _image(entry.frame)]
        content += [{"type": "text", "text": "Current screenshot:"}, _image(frame)]
        payload = {
            **self._settings.request,
            "model": self._settings.name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": field, "strict": True,
                "schema": {"type": "object", "properties": {field: schema},
                           "required": [field], "additionalProperties": False},
            }},
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
            return json.loads(data["choices"][0]["message"]["content"])[field]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{self.name} returned an invalid answer") from exc


def _image(frame: Frame) -> dict:
    return {"type": "image_url", "image_url": {"url": frame.data_url}}
