"""The review of a task's turns: facts about the game for long-term memory."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol, get_args

from mystery_dungeon.actions import name
from mystery_dungeon.model import (
    ChatSettings,
    api_key,
    decode,
    describe,
    image,
    nearby,
    one_line,
    post,
)

if TYPE_CHECKING:
    from mystery_dungeon.agent import Turn

# Each request reviews this many turns, so its size and answer stay bounded on long tasks. On
# the plaza dialogue cases, 20-turn chunks recalled as many dialogue boxes as whole tasks did
# (68 of 72).
REVIEW_TURNS = 20
# A chunk's answer copies every dialogue box in full, far longer than an action; the plaza
# reviews ran with this budget.
REVIEW_MAX_TOKENS = 8192
REVIEW_WORKERS = 4  # chunks in flight at once, so a long task doesn't hit rate limits
FindingKind = Literal["dialogue", "location", "mechanic", "other"]
FINDING_KINDS: tuple[str, ...] = get_args(FindingKind)
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array", "items": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": list(FINDING_KINDS)},
                       "text": {"type": "string"}},
        "required": ["kind", "text"],
        "additionalProperties": False}}},
    "required": ["findings"],
    "additionalProperties": False,
}
# The game state's names keep the review from inventing characters it misreads on screen.
REVIEW_INSTRUCTIONS = """\
You review part of a playthrough of Pokémon Mystery Dungeon: Red Rescue Team. You get the \
task and each turn: the screenshot, the game state read from the game (the names and positions \
of the characters on screen are exact), the player's reasoning and the action taken. Write \
findings: facts about the game a later player could use.
- dialogue: every distinct dialogue box once, with its speaker and its full text, copied from \
the screenshots. A box whose text grows over several screenshots is one box: give its \
complete text once.
- location: where a named character or place is, relative to named landmarks.
- mechanic: how the game responded to an action, only when the turns show it clearly; say how \
often it was seen.
- other: other on-screen text, such as menu choices or prices.
Name characters only as the game state names them. Leave out the player's progress, plans and \
guesses, and anything the screenshots don't show."""


@dataclass(frozen=True)
class Finding:
    """A fact about the game, written to stand alone."""

    kind: FindingKind
    text: str


@dataclass(frozen=True)
class ReviewError:
    """A review that failed."""

    message: str


class Reviewer(Protocol):
    """Find facts about the game in a task's turns."""

    name: str

    def review(self, task: str, turns: Sequence[Turn]) -> list[Finding]:
        """Return the findings in the turns, in turn order.

        Raises:
            Exception: when a review answer is missing or invalid.
        """
        ...


def review(reviewer: Reviewer, task: str, turns: Sequence[Turn]) -> list[Finding] | ReviewError:
    """Review the turns, converting a failure into a ReviewError."""
    try:
        return reviewer.review(task, turns)
    except Exception as exc:
        return ReviewError(f"{type(exc).__name__}: {exc}")


class ChatReviewer:
    """A reviewer behind the same chat-completions endpoint as the model."""

    def __init__(self, settings: ChatSettings) -> None:
        self._settings = replace(settings, max_tokens=REVIEW_MAX_TOKENS)
        self._api_key = api_key(settings)
        self.name = settings.name

    def review(self, task: str, turns: Sequence[Turn]) -> list[Finding]:
        starts = range(0, len(turns), REVIEW_TURNS)
        with ThreadPoolExecutor(REVIEW_WORKERS) as pool:
            parts = list(pool.map(
                lambda start: self._review(task, turns[start:start + REVIEW_TURNS], start),
                starts))
        return [finding for part in parts for finding in part]

    def _review(self, task: str, turns: Sequence[Turn], first: int) -> list[Finding]:
        response = post(self._settings, self._api_key, REVIEW_INSTRUCTIONS,
                        _content(task, turns, first), REVIEW_SCHEMA)
        answer = decode(response, self.name)
        if not isinstance(answer, dict) or answer.keys() != {"findings"} or not isinstance(
                answer["findings"], list):
            raise ValueError(f"{self.name} returned an invalid review: {answer!r}")
        return [_finding(item, self.name) for item in answer["findings"]]


def _finding(item: object, model: str) -> Finding:
    if (not isinstance(item, dict) or item.keys() != {"kind", "text"}
            or item["kind"] not in FINDING_KINDS or not isinstance(item["text"], str)
            or not item["text"].strip()):
        raise ValueError(f"{model} returned an invalid finding: {item!r}")
    return Finding(item["kind"], one_line(item["text"]))


def _content(task: str, turns: Sequence[Turn], first: int) -> list[dict]:
    content = [{"type": "text", "text": f"# Task\n{task}"}]
    for number, turn in enumerate(turns, first + 1):
        state = turn.observation.game_state
        content += [{"type": "text", "text": f"## Turn {number}"},
                    image(turn.observation.frame),
                    {"type": "text", "text": "\n".join([
                        f"Game state: {describe(state)}", *nearby(state),
                        f"Reasoning: {turn.reasoning}", f"Action: {name(turn.action)}"])}]
    return content
