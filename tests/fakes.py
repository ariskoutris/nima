"""Test doubles: a scripted model, a fake emulator, and helpers to drive them."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Sequence

from emulator import Frame
from mystery_dungeon.actions import GAME_ACTIONS, Stop
from mystery_dungeon.model import Reply
from mystery_dungeon.session import HOLD_TICKS, RELEASE_TICKS, GameplaySession, perform

FRAME = Frame(16, 16, bytes(16 * 16 * 4))  # large enough for stall detection to sample


class ScriptedModel:
    """Answer from fixed scripts and record every call.

    Actions are model-facing names, and the last one repeats; notes go with the actions in
    order and run out. Each call replies Stop, with the reasoning as its reason, instead when
    ``stop()`` returns True, and raises ``error`` when one is given. Reports add the reasoning
    to a fixed text.
    With a ``gate``, each call sets ``entered`` and then waits for the gate.
    """

    def __init__(self, actions: Sequence[str] = ("A",),
                 stop: Callable[[], bool] = lambda: False, *,
                 gate: threading.Event | None = None, error: Exception | None = None,
                 reasoning: str = "Scripted.", notes: Sequence[str | None] = ()) -> None:
        self.actions = list(actions)
        self.notes = list(notes)
        self.reasoning = reasoning
        self.stop = stop
        self.gate = gate
        self.error = error
        self.entered = threading.Event()
        self.calls: list[tuple] = []
        self.name = "scripted"
        self.last_response: dict | None = None
        self._chosen = 0

    def call(self, context, frame) -> Reply:
        self._called("call", context, frame)
        if self.stop():
            return Reply(Stop(self.reasoning), self.reasoning, self.reasoning)
        action = self.actions[min(self._chosen, len(self.actions) - 1)]
        note = self.notes[self._chosen] if self._chosen < len(self.notes) else None
        self._chosen += 1
        return Reply(GAME_ACTIONS[action], self.reasoning, note)

    def report(self, context, frame) -> str:
        self._called("report", context, frame)
        return f"Out of budget. {self.reasoning}"

    def _called(self, *call) -> None:
        self.calls.append(call)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(2), "test gate was never released"
        if self.error is not None:
            raise self.error


def votes(*answers: bool) -> Callable[[], bool]:
    """Stop on the given calls in order, then keep acting."""
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


def rows(path: Path) -> list[dict]:
    """Return the rows of a JSON Lines log, or none before the first row."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def press(game, action: str) -> None:
    """Perform a named action on a real emulator with the benchmark's timing."""
    perform(game, GAME_ACTIONS[action], HOLD_TICKS, RELEASE_TICKS)
