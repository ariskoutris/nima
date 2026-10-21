"""The model the agent asks for each action, and its chat-completions implementation."""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, get_args
from urllib.request import Request, urlopen

from emulator import Frame
from mystery_dungeon import CONFIG
from mystery_dungeon.actions import ACTION_NAMES, ACTIONS, Action, name
from mystery_dungeon.prompts import ACTION_DESCRIPTIONS, INSTRUCTIONS

if TYPE_CHECKING:
    from mystery_dungeon.agent import Context
    from mystery_dungeon.session import GameState

TIMEOUT_SECONDS = 120
# Frames go to the model enlarged; at native 240x160 it misread the shop menu cursor.
IMAGE_SCALE = 2
FindingKind = Literal["dialogue", "location", "mechanic", "other"]
NUMBERED = re.compile(r"\[\d+\]")  # a scratchpad line number as the context shows it
# The model judges the screen, edits its scratchpad and records findings before it commits to
# an action.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "scratchpad": {"type": "object", "properties": {
            "rewrite": {"type": ["array", "null"], "items": {"type": "string"}},
            "lines": {"type": "array", "items": {
                "type": "object", "properties": {"line": {"type": "integer"},
                                                 "text": {"type": ["string", "null"]}},
                "required": ["line", "text"], "additionalProperties": False}},
            "add": {"type": "array", "items": {"type": "string"}}},
            "required": ["rewrite", "lines", "add"], "additionalProperties": False},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": list(get_args(FindingKind))},
                           "text": {"type": "string"}},
            "required": ["kind", "text"], "additionalProperties": False}},
        "action": {"type": "string", "enum": list(ACTION_NAMES)},
    },
    "required": ["reasoning", "scratchpad", "findings", "action"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LineChange:
    """New text for a scratchpad line, by the number shown this turn; None deletes it."""

    line: int
    text: str | None


@dataclass(frozen=True)
class ScratchpadEdit:
    """Either a rewrite of the whole scratchpad, or changes to shown lines plus lines appended
    at the end."""

    rewrite: tuple[str, ...] | None = None
    lines: tuple[LineChange, ...] = ()
    add: tuple[str, ...] = ()

    def problem(self, shown: int) -> str | None:
        """Say why the edit can't apply to a scratchpad of ``shown`` lines, or None if it can."""
        if self.rewrite is not None and (self.lines or self.add):
            return "a rewrite came with line changes or additions"
        numbers = [change.line for change in self.lines]
        if len(set(numbers)) != len(numbers):
            twice = sorted({number for number in numbers if numbers.count(number) > 1})
            return f"lines {twice} were changed more than once"
        missing = sorted(number for number in numbers if not 1 <= number <= shown)
        if missing:
            return f"lines {missing} are not shown"
        texts = [*(self.rewrite or ()), *self.add,
                 *(change.text for change in self.lines if change.text is not None)]
        if not all(text.strip() for text in texts):
            return "a line was empty; delete a line with null instead"
        # The model copied shown numbers like "[4]" into its text, then edited by those.
        if any(NUMBERED.match(text.strip()) for text in texts):
            return "a line's text started with a line number"
        return None


@dataclass(frozen=True)
class Finding:
    """A standalone fact about the game; any uncertainty is stated in its text."""

    kind: FindingKind
    text: str


@dataclass(frozen=True)
class Reply:
    """The model's action, its reasoning, its scratchpad edit and its findings (often none)."""

    action: Action
    reasoning: str
    scratchpad: ScratchpadEdit
    findings: tuple[Finding, ...]


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
        self._api_key = os.getenv(settings.api_key_env) if settings.api_key_env else None
        if settings.api_key_env and not self._api_key:
            raise ValueError(f"{settings.api_key_env} is required for {settings.name}")
        self.name = settings.name
        self.last_response: dict | None = None

    def call(self, context: Context, frame: Frame) -> Reply:
        answer = self._send(_content(context, frame))
        if not isinstance(answer, dict) or answer.keys() != ANSWER_SCHEMA["properties"].keys():
            raise ValueError(f"{self.name} returned an invalid answer: {answer!r}")
        if not isinstance(answer["action"], str) or answer["action"] not in ACTION_NAMES:
            raise ValueError(f"{self.name} returned an unknown action: {answer['action']!r}")
        if not isinstance(answer["reasoning"], str) or not answer["reasoning"].strip():
            raise ValueError(f"{self.name} returned no reasoning: {answer['reasoning']!r}")
        if not isinstance(answer["findings"], list):
            raise ValueError(f"{self.name} returned invalid findings: {answer['findings']!r}")
        return Reply(ACTIONS[answer["action"]], answer["reasoning"].strip(),
                     self._edit(answer["scratchpad"]),
                     tuple(self._finding(finding) for finding in answer["findings"]))

    def _edit(self, edit: object) -> ScratchpadEdit:
        """Check the edit's shape; ScratchpadEdit.problem judges whether it can apply."""
        if (not isinstance(edit, dict) or edit.keys() != {"rewrite", "lines", "add"}
                or not isinstance(edit["rewrite"], list | None)
                or not isinstance(edit["lines"], list) or not isinstance(edit["add"], list)
                or not all(isinstance(change, dict) and change.keys() == {"line", "text"}
                           and type(change["line"]) is int
                           and isinstance(change["text"], str | None)
                           for change in edit["lines"])
                or not all(isinstance(text, str)
                           for text in [*(edit["rewrite"] or ()), *edit["add"]])):
            raise ValueError(f"{self.name} returned an invalid scratchpad edit: {edit!r}")
        return ScratchpadEdit(
            # An empty rewrite means no rewrite; clearing every line would lose the scratchpad.
            tuple(text.strip() for text in edit["rewrite"]) if edit["rewrite"] else None,
            tuple(LineChange(change["line"], None if change["text"] is None
                             else change["text"].strip()) for change in edit["lines"]),
            tuple(text.strip() for text in edit["add"]))

    def _finding(self, finding: object) -> Finding:
        if (not isinstance(finding, dict) or finding.keys() != {"kind", "text"}
                or finding["kind"] not in get_args(FindingKind)
                or not isinstance(finding["text"], str) or not finding["text"].strip()):
            raise ValueError(f"{self.name} returned an invalid finding: {finding!r}")
        return Finding(finding["kind"], finding["text"].strip())

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
    """Build the user message: task, knowledge, scratchpad and actions, earlier turns, current
    frame."""
    text = [f"# Task\n{context.task}"]
    if context.knowledge:
        text.append(f"# Knowledge\n{context.knowledge}")
    text.append("# Your scratchpad\n" + ("\n".join(
        f"[{number}] {line}" for number, line in enumerate(context.scratchpad, 1)) or "(empty)"))
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
        content += [{"type": "text", "text": heading}, _image(entry.observation.frame),
                    {"type": "text", "text": "\n".join(after)}]
    current = [{"type": "text", "text": "# Current screenshot"}, _image(frame)]
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
    lines = ["Characters on screen, relative to you:"]
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
