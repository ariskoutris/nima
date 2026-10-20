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
    from mystery_dungeon.session import GameState

TIMEOUT_SECONDS = 120
# Frames go to the model enlarged; at native 240x160 it misread the shop menu cursor.
IMAGE_SCALE = 2
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


def internal_reasoning(response: dict | None) -> str | None:
    """Return the reasoning a chat response carries besides its answer, if any."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    # OpenRouter names the field "reasoning"; vLLM and others name it "reasoning_content".
    return message.get("reasoning") or message.get("reasoning_content")


def _content(context: Context, frame: Frame) -> list[dict]:
    """Build the user message: task, knowledge and actions, earlier turns, current frame."""
    text = [f"# Task\n{context.task}"]
    if context.knowledge:
        text.append(f"# Knowledge\n{context.knowledge}")
    text.append("# Actions\n" + "\n".join(
        f"- {action}: {description}" for action, description in ACTION_DESCRIPTIONS.items()))
    if context.memory:
        text.append("# Earlier turns (oldest first)")
    content = [{"type": "text", "text": "\n\n".join(text)}]
    for age, entry in zip(range(len(context.memory), 0, -1), context.memory, strict=True):
        after = [] if entry.observation.game_state is None else [
            f"Game state: {describe(entry.observation.game_state)}"]
        if entry.reasoning is None:
            heading = f"## Turn -{age} (operator)"
        else:
            heading = f"## Turn -{age}"
            after.append(f"Reasoning: {entry.reasoning}")
        after.append(f"Action: {name(entry.action)}")
        content += [{"type": "text", "text": heading}, _image(entry.observation.frame),
                    {"type": "text", "text": "\n".join(after)}]
    current = [{"type": "text", "text": "# Current screenshot"}, _image(frame)]
    if context.game_state is not None:
        current.append({"type": "text", "text": f"# Game state\n{describe(context.game_state)}"})
    return content + current


def describe(state: GameState) -> str:
    """Describe a game state in one line, leaving out what isn't known."""
    parts = []
    if state.position is not None:
        parts.append(f"player position {state.position} pixels")
    if state.tile is not None:
        parts.append(f"player tile {state.tile}")
    if state.direction is not None:
        parts.append(f"facing {state.direction}")
    return ", ".join(parts) or "unknown"


def _image(frame: Frame) -> dict:
    return {"type": "image_url", "image_url": {"url": scaled(frame, IMAGE_SCALE).data_url}}


def scaled(frame: Frame, factor: int) -> Frame:
    """Enlarge a frame by a whole factor with nearest-neighbour sampling."""
    row_bytes = frame.width * 4
    rows = []
    for top in range(0, len(frame.rgba), row_bytes):
        row = frame.rgba[top:top + row_bytes]
        wide = b"".join(row[left:left + 4] * factor for left in range(0, row_bytes, 4))
        rows.append(wide * factor)
    return Frame(frame.width * factor, frame.height * factor, b"".join(rows))
