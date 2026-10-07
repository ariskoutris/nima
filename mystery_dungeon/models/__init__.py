"""Contracts for planning and action models, and loading a model by name."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from emulator import Frame
    from mystery_dungeon.actions import Action
    from mystery_dungeon.agent import MemoryEntry


class ActionModel(Protocol):
    """Chooses the next action for a task."""

    last_response: dict | None  # raw answer to the latest call, kept for the run log

    def choose_action(self, task: str, knowledge: str, frame: Frame,
                      memory: tuple[MemoryEntry, ...]) -> Action:
        """Return one action from ACTIONS for the current frame.

        Args:
            memory: recent screenshots, oldest first, each with the action taken after it.

        Raises:
            Exception: when the model fails or answers outside the action space.
        """
        ...


class PlanningModel(Protocol):
    """Judges progress on a task."""

    last_response: dict | None

    def is_complete(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...]) -> bool:
        """Judge from visible evidence only whether the task has been completed.

        Raises:
            Exception: when the model fails or does not answer yes or no.
        """
        ...


class Model(ActionModel, PlanningModel, Protocol):
    """One model filling both roles."""


def load_model(spec: str) -> Model:
    """Instantiate MODULE:CLASS with no arguments.

    Raises:
        ValueError: when spec is not MODULE:CLASS.
        TypeError: when the class does not provide both model roles.
    """
    module, separator, class_name = spec.partition(":")
    if not (module and separator and class_name):
        raise ValueError("model must be MODULE:CLASS")
    cls = getattr(importlib.import_module(module), class_name)
    if not isinstance(cls, type) or not all(
            callable(getattr(cls, method, None)) for method in ("choose_action", "is_complete")):
        raise TypeError(f"{spec} must be a class with choose_action and is_complete")
    return cls()
