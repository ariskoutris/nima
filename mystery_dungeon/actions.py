"""The agent's action space."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Press:
    """Hold one GBA button for an action's hold time, then release it."""

    button: str


@dataclass(frozen=True)
class Wait:
    """Press nothing and let the game advance for one action's duration."""


@dataclass(frozen=True)
class Done:
    """Report the task complete; it goes to the runner, never to the game."""


GameAction = Press | Wait  # what the game performs
Action = GameAction | Done

GAME_ACTIONS: dict[str, GameAction] = {
    **{button: Press(button) for button in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "SELECT")},
    "WAIT": Wait(),
}
ACTIONS: dict[str, Action] = {**GAME_ACTIONS, "DONE": Done()}
_NAMES = {action: name for name, action in ACTIONS.items()}


def name(action: Action) -> str:
    """Return the model-facing name of an action in ACTIONS."""
    return _NAMES[action]


def to_json(action: Action) -> dict[str, str]:
    return {"type": type(action).__name__.lower(), **asdict(action)}
