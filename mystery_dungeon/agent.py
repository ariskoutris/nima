"""Agent that interacts with the game to complete the given task."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mystery_dungeon.actions import Action, Done, GameAction

if TYPE_CHECKING:
    from mystery_dungeon.model import Model
    from mystery_dungeon.session import GameplaySession, Observation, Transition

MEMORY_TURNS = 5  # not yet tuned


@dataclass(frozen=True)
class MemoryEntry:
    """An earlier turn. Reasoning is None when the operator acted."""

    observation: Observation
    action: GameAction
    reasoning: str | None


@dataclass(frozen=True)
class Context:
    """Everything the model gets besides the current frame."""

    task: str
    knowledge: str
    memory: tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class Turn:
    observation: Observation
    context: Context
    action: Action
    reasoning: str


@dataclass(frozen=True)
class ModelError:
    """A turn where the model call failed."""

    observation: Observation
    context: Context
    message: str


class Agent:
    """Agent that interacts with the game to complete the given task.
    Chooses actions with a model and keeps recent turns in memory."""

    def __init__(self, model: Model, knowledge: str = "") -> None:
        self._model = model
        self._knowledge = knowledge.strip()
        self._task = ""
        self._memory: deque[MemoryEntry] = deque(maxlen=MEMORY_TURNS)

    @property
    def memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._memory)

    def set_task(self, task: str) -> None:
        """Switch tasks, forgetting the old task's memory."""
        task = task.strip()
        if self._task and task != self._task:
            self._memory.clear()
        self._task = task

    def clear_memory(self) -> None:
        self._memory.clear()

    def act(self, session: GameplaySession) -> tuple[Turn | ModelError, Transition | None]:
        """Take one turn: observe, ask the model and apply its action."""
        if not self._task:
            raise ValueError("give the agent a task before it acts")
        observation = session.observe()
        context = Context(self._task, self._knowledge, self.memory)
        try:
            reply = self._model.call(context, observation.frame)
        except Exception as exc:
            return ModelError(observation, context, f"{type(exc).__name__}: {exc}"), None
        turn = Turn(observation, context, reply.action, reply.reasoning)
        match reply.action:
            case Done():
                return turn, None
            case action:
                return turn, self._apply(session, action, reply.reasoning)

    def manual(self, session: GameplaySession, action: GameAction) -> Transition:
        """Apply an operator's action."""
        return self._apply(session, action, None)

    def _apply(self, session: GameplaySession, action: GameAction,
               reasoning: str | None) -> Transition:
        transition = session.apply(action)
        self._memory.append(MemoryEntry(transition.before, action, reasoning))
        return transition
