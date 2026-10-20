"""Contracts for planning and action models."""

from __future__ import annotations

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
