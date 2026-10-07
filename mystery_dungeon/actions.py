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


Action = Press | Wait

ACTIONS: dict[str, Action] = {
    **{button: Press(button) for button in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "SELECT")},
    "WAIT": Wait(),
}
_NAMES = {action: name for name, action in ACTIONS.items()}


def name(action: Action) -> str:
    """Return the model-facing name of an action in ACTIONS."""
    return _NAMES[action]


def to_json(action: Action) -> dict[str, str]:
    return {"type": type(action).__name__.lower(), **asdict(action)}
