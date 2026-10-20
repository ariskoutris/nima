"""Decide what the agent does next: plan a task, then choose an action for it."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from emulator import Frame
from mystery_dungeon.actions import Action

if TYPE_CHECKING:
    from mystery_dungeon.model import ActionModel, PlanningModel
    from mystery_dungeon.session import GameplaySession, Observation, Transition

MEMORY_ENTRIES = 1


@dataclass(frozen=True)
class MemoryEntry:
    """A screenshot and the action taken right after it."""

    frame: Frame
    screenshot: str
    action: Action


@dataclass(frozen=True)
class Task:
    description: str
    knowledge: str


@dataclass(frozen=True)
class Turn:
    """What the agent proposed for one observation; nothing has been applied yet."""

    observation: Observation
    task: Task
    memory: tuple[MemoryEntry, ...]
    complete: bool | None = None
    action: Action | None = None
    error: str | None = None

    @property
    def status(self) -> Literal["model_error", "completion", "action"]:
        if self.error is not None:
            return "model_error"
        return "completion" if self.complete else "action"


class Knowledge:
    """Experience the agent carries across tasks; only seeded text for now."""

    def __init__(self, seed: str) -> None:
        self._entries = [seed.strip()] if seed.strip() else []

    def relevant(self, task: str) -> str:
        """Return the knowledge that bears on a task (currently all of it)."""
        return "\n\n".join(self._entries)


class Agent:
    """Plan which task to pursue and choose actions for it.

    The agent touches the game only through commit() and manual(); everything
    else works from observations.
    """

    def __init__(self, planning_model: PlanningModel, action_model: ActionModel,
                 knowledge: str = "") -> None:
        self._planning_model = planning_model
        self._action_model = action_model
        self._knowledge = Knowledge(knowledge)
        self._instructions = ""
        self._task: Task | None = None
        self._memory: deque[MemoryEntry] = deque(maxlen=MEMORY_ENTRIES)

    @property
    def task(self) -> Task | None:
        return self._task

    @property
    def memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._memory)

    def set_instructions(self, instructions: str) -> None:
        self._instructions = instructions.strip()

    def clear_memory(self) -> None:
        self._memory.clear()

    def plan(self, observation: Observation) -> tuple[Task, bool]:
        """Choose the task to pursue and judge whether it is already complete.

        Knowledge reconciliation and reprioritization belong here; for now the
        task is always the operator's instructions.

        Raises:
            ValueError: when the agent has no instructions.
            Exception: whatever the planning model raises.
        """
        if not self._instructions:
            raise ValueError("give the agent instructions before planning")
        if self._task is None or self._task.description != self._instructions:
            self._task = Task(self._instructions, self._knowledge.relevant(self._instructions))
        complete = self._planning_model.is_complete(self._task.description, observation.frame,
                                                    self.memory)
        return self._task, complete

    def act(self, observation: Observation, task: Task) -> Action:
        """Choose the next action for a task.

        Raises:
            Exception: whatever the action model raises.
        """
        return self._action_model.choose_action(task.description, task.knowledge,
                                                observation.frame, self.memory)

    def propose(self, observation: Observation) -> Turn:
        """Plan, then choose an action unless the task is complete, without touching the game.

        Model failures are returned as the turn's error instead of being raised.

        Raises:
            ValueError: when the agent has no instructions.
        """
        if not self._instructions:
            raise ValueError("give the agent instructions before planning")
        memory = self.memory
        try:
            task, complete = self.plan(observation)
            action = None if complete else self.act(observation, task)
        except Exception as exc:
            return Turn(observation, self._task, memory, error=f"{type(exc).__name__}: {exc}")
        return Turn(observation, task, memory, complete, action)

    def commit(self, session: GameplaySession, turn: Turn) -> Transition | None:
        """Apply a proposed action and remember it; turns without an action change nothing."""
        if turn.action is None:
            return None
        return self._apply(session, turn.action)

    def manual(self, session: GameplaySession, action: Action) -> Transition:
        """Apply an operator's action; it enters memory like the agent's own."""
        return self._apply(session, action)

    def take_turn(self, session: GameplaySession) -> tuple[Turn, Transition | None]:
        """Observe, propose and commit in one call for single-threaded runners."""
        turn = self.propose(session.observe())
        return turn, self.commit(session, turn)

    def _apply(self, session: GameplaySession, action: Action) -> Transition:
        transition = session.apply(action)
        self._memory.append(MemoryEntry(transition.before, transition.before_name, action))
        return transition
