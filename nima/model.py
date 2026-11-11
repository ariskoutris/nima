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
from nima import CONFIG
from nima.actions import ACTION_NAMES, ACTIONS, Action, name
from nima.prompts import ACTION_DESCRIPTIONS, INSTRUCTIONS

if TYPE_CHECKING:
    from nima.agent import Context
    from nima.session import GameState

TIMEOUT_SECONDS = 120
# Frames go to the model enlarged; at native 240x160 it misread the shop menu cursor.
IMAGE_SCALE = 2
# The model judges the screen and may add a note before it commits to an action.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "note": {"type": "string"},
        "action": {"type": "string", "enum": list(ACTION_NAMES)},
    },
    "required": ["reasoning", "note", "action"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Reply:
    """The model's action and reasoning, and a note to keep (None for none)."""

    action: Action
    reasoning: str
    note: str | None


class Model(Protocol):
    """Choose the next action from the context and current frame."""

    name: str
    last_response: dict | None  # raw response to the latest call, for the run log

    def call(self, context: Context, frame: Frame) -> Reply:
        """Return the next action, or Stop with how the task ended.

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
        self._api_key = api_key(settings)
        self.name = settings.name
        self.last_response: dict | None = None

    def call(self, context: Context, frame: Frame) -> Reply:
        self.last_response = None
        self.last_response = post(self._settings, self._api_key, INSTRUCTIONS,
                                  _content(context, frame), ANSWER_SCHEMA)
        answer = decode(self.last_response, self.name)
        if not isinstance(answer, dict) or answer.keys() != ANSWER_SCHEMA["properties"].keys():
            raise ValueError(f"{self.name} returned an invalid answer: {answer!r}")
        if not isinstance(answer["action"], str) or answer["action"] not in ACTION_NAMES:
            raise ValueError(f"{self.name} returned an unknown action: {answer['action']!r}")
        if not isinstance(answer["reasoning"], str) or not answer["reasoning"].strip():
            raise ValueError(f"{self.name} returned no reasoning: {answer['reasoning']!r}")
        if not isinstance(answer["note"], str):
            raise ValueError(f"{self.name} returned an invalid note: {answer['note']!r}")
        return Reply(ACTIONS[answer["action"]], answer["reasoning"].strip(),
                     one_line(answer["note"]) or None)


def api_key(settings: ChatSettings) -> str | None:
    """The key named by settings.api_key_env, or None when the endpoint needs none.

    Raises:
        ValueError: when the named variable is unset.
    """
    if not settings.api_key_env:
        return None
    key = os.getenv(settings.api_key_env)
    if not key:
        raise ValueError(f"{settings.api_key_env} is required for {settings.name}")
    return key


def post(settings: ChatSettings, key: str | None, system: str, content: list[dict],
         schema: dict) -> dict:
    """Send one chat request whose answer must match a strict JSON schema; return the raw
    response."""
    payload = {
        **settings.request,
        "model": settings.name,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "answer", "strict": True, "schema": schema}},
        "max_tokens": settings.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(settings.url, data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def decode(response: dict, model: str) -> object:
    """Return a chat response's answer as decoded JSON.

    Raises:
        ValueError: when the answer was cut off or isn't JSON.
    """
    if response["choices"][0].get("finish_reason") == "length":
        raise ValueError(f"{model} exhausted max_tokens before answering")
    try:
        return json.loads(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{model} returned an invalid answer") from exc


def one_line(text: str) -> str:
    """Collapse whitespace, newlines included, to single spaces."""
    return " ".join(text.split())


def internal_reasoning(response: dict | None) -> str | None:
    """Return the reasoning a chat response carries besides its answer, if any."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    # OpenRouter names the field "reasoning"; vLLM and others name it "reasoning_content".
    return message.get("reasoning") or message.get("reasoning_content")


def token_usage(response: dict | None) -> dict | None:
    """Return a response's tokens as prompt, output and reasoning, or None if unreported.

    Output excludes reasoning, so the three sum to the total. Reasoning is None when the
    endpoint doesn't report it.
    """
    usage = (response or {}).get("usage")
    if not usage:
        return None
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {"prompt": usage["prompt_tokens"],
            "output": usage["completion_tokens"] - (reasoning or 0),
            "reasoning": reasoning}


def _content(context: Context, frame: Frame) -> list[dict]:
    """Build the user message: task, knowledge, notes and actions, earlier turns,
    current frame."""
    text = [f"# Task\n{context.task}"]
    if context.knowledge:
        text.append(f"# Knowledge\n{context.knowledge}")
    text.append("# Your notes\n" + ("\n".join(
        f"- {note}" for note in context.notes) or "(none yet)"))
    text.append("# Actions\n" + "\n".join(
        f"- {action}: {description}" for action, description in ACTION_DESCRIPTIONS.items()))
    if context.memory:
        text.append("# Earlier turns (oldest first)")
    content = [{"type": "text", "text": "\n\n".join(text)}]
    for age, entry in zip(range(len(context.memory), 0, -1), context.memory, strict=True):
        after = [f"Game state: {describe(entry.observation.game_state)}"]
        if entry.reasoning is None:
            heading = f"## Turn -{age} (operator)"
        else:
            heading = f"## Turn -{age}"
            after.append(f"Reasoning: {entry.reasoning}")
        after.append(f"Action: {name(entry.action)}")
        content += [{"type": "text", "text": heading}, image(entry.observation.frame),
                    {"type": "text", "text": "\n".join(after)}]
    current = [{"type": "text", "text": "# Current screenshot"}, image(frame)]
    state = context.game_state
    you = [] if state.species is None else [f"You play as {state.species}."]
    current.append({"type": "text", "text": "# Game state\n" + "\n".join(
        [*you, describe(state), *nearby(state)])})
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


def nearby(state: GameState) -> list[str]:
    """Describe each on-screen character by its offset from the player."""
    if state.position is None or not state.characters:
        return []
    # Without the unit the model sometimes read offsets as tiles ("1 right, 2 up").
    lines = ["Characters on screen, offset from you in pixels:"]
    for character in state.characters:
        offset = ", ".join(part for part in (
            _offset(character.position[0] - state.position[0], "right", "left"),
            _offset(character.position[1] - state.position[1], "down", "up")) if part)
        lines.append(f"- {character.name}: {offset or 'same position'}, "
                     f"facing {character.direction}")
    return lines


def _offset(distance: int, positive: str, negative: str) -> str:
    if distance == 0:
        return ""
    return f"{abs(distance)} {positive if distance > 0 else negative}"


def image(frame: Frame) -> dict:
    """A frame as an image part of a chat message, enlarged by IMAGE_SCALE."""
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
