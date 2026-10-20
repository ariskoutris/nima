"""Test doubles: a scripted model, a fake emulator, and helpers to drive them."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Sequence

from emulator import Frame
from mystery_dungeon.actions import ACTIONS, GAME_ACTIONS, Done
from mystery_dungeon.model import Reply
from mystery_dungeon.session import HOLD_TICKS, RELEASE_TICKS, GameplaySession, perform

FRAME = Frame(16, 16, bytes(16 * 16 * 4))  # large enough for stall detection to sample


class ScriptedModel:
    """Answer from fixed scripts and record every call.

    Actions are model-facing names, and the last one repeats. Each call replies Done
    instead when ``complete()`` returns True, and raises ``error`` when one is given.
    With a ``gate``, each call sets ``entered`` and then waits for the gate.
    """

    def __init__(self, actions: Sequence[str] = ("A",),
                 complete: Callable[[], bool] = lambda: False, *,
                 gate: threading.Event | None = None, error: Exception | None = None,
                 reasoning: str = "Scripted.") -> None:
        self.actions = list(actions)
        self.reasoning = reasoning
        self.complete = complete
        self.gate = gate
        self.error = error
        self.entered = threading.Event()
        self.calls: list[tuple] = []
        self.name = "scripted"
        self.last_response: dict | None = None
        self._chosen = 0

    def call(self, context, frame) -> Reply:
        self._called("call", context, frame)
        if self.complete():
            return Reply(Done(), self.reasoning)
        action = self.actions[min(self._chosen, len(self.actions) - 1)]
        self._chosen += 1
        return Reply(ACTIONS[action], self.reasoning)

    def _called(self, *call) -> None:
        self.calls.append(call)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(2), "test gate was never released"
        if self.error is not None:
            raise self.error


def votes(*answers: bool) -> Callable[[], bool]:
    """Report completion on the given calls in order, then keep acting."""
    remaining = iter(answers)
    return lambda: next(remaining, False)


class Game:
    """An emulator stand-in that records inputs and keeps state and save RAM as bytes."""

    def __init__(self) -> None:
        self.frame = FRAME
        self.steps: list[tuple[int, tuple[str, ...]]] = []
        self.state = b"current"
        self.ram = b"current-ram"
        self.on_frame = None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def screenshot(self) -> Frame:
        return self.frame

    def step(self, ticks: int = 1, buttons: tuple[str, ...] = ()) -> None:
        self.steps.append((ticks, tuple(buttons)))
        for _ in range(ticks if self.on_frame else 0):
            self.on_frame(self.frame)

    def save_state(self) -> bytes:
        return self.state

    def load_state(self, data: bytes) -> None:
        if data == b"bad":
            raise RuntimeError("invalid core state")
        self.state = data

    def save_ram(self) -> bytes:
        return self.ram

    def load_save_ram(self, data: bytes) -> None:
        self.ram = data


def start(tmp_path, game=None) -> GameplaySession:
    """Return a session over ``game`` (a fresh Game by default) with its first frame captured."""
    session = GameplaySession(game or Game(), tmp_path)
    session.capture()
    return session


def events(output: Path) -> list[dict]:
    """Rows of a run's events.jsonl, or none before the first event."""
    path = output / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def press(game, action: str) -> None:
    """Perform a named action on a real emulator with the benchmark's timing."""
    perform(game, GAME_ACTIONS[action], HOLD_TICKS, RELEASE_TICKS)
