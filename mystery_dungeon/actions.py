"""The agent's action space."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, get_args

Outcome = Literal["complete", "stuck", "invalid"]


@dataclass(frozen=True)
class Press:
    """Hold one GBA button for an action's hold time, then release it."""

    button: str


@dataclass(frozen=True)
class Wait:
    """Press nothing and let the game advance for one action's duration."""


@dataclass(frozen=True)
class Stop:
    """End the attempt, claiming how it ended; it goes to the runner, never to the game."""

    outcome: Outcome


GameAction = Press | Wait  # what the game performs
Action = GameAction | Stop

GAME_ACTIONS: dict[str, GameAction] = {
    **{button: Press(button) for button in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "SELECT")},
    "WAIT": Wait(),
}
STOPS: dict[str, Stop] = {f"STOP_{outcome.upper()}": Stop(outcome)
                          for outcome in get_args(Outcome)}
ACTIONS: dict[str, Action] = {**GAME_ACTIONS, **STOPS}  # by model-facing name
ACTION_NAMES = tuple(ACTIONS)
_NAMES = {action: name for name, action in ACTIONS.items()}


def name(action: Action) -> str:
    """Return the model-facing name of an action."""
    return _NAMES[action]


def to_json(action: Action) -> dict[str, str]:
    return {"type": type(action).__name__.lower(), **asdict(action)}
