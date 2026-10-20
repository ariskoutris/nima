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
class Stop:
    """End the attempt and say why; it goes to the runner, never to the game."""

    reason: str


GameAction = Press | Wait  # what the game performs
Action = GameAction | Stop

GAME_ACTIONS: dict[str, GameAction] = {
    **{button: Press(button) for button in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "SELECT")},
    "WAIT": Wait(),
}
ACTION_NAMES = (*GAME_ACTIONS, "STOP")  # model-facing
_NAMES = {action: name for name, action in GAME_ACTIONS.items()}


def name(action: Action) -> str:
    """Return the model-facing name of an action."""
    match action:
        case Stop():
            return "STOP"
        case _:
            return _NAMES[action]


def to_json(action: Action) -> dict[str, str]:
    return {"type": type(action).__name__.lower(), **asdict(action)}
