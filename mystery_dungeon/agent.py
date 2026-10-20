"""Pursue one task: act on the current frame and recent turns until the model reports DONE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mystery_dungeon.actions import Action, Done, GameAction

if TYPE_CHECKING:
    from mystery_dungeon.model import Model
    from mystery_dungeon.session import GameplaySession, Observation, Transition

MEMORY_TURNS = 5  # earlier turns every model call sees; not yet tuned


@dataclass(frozen=True)
class MemoryEntry:
    """What the agent saw, the action taken right after it, and the reasoning behind it.

    Reasoning is None for operator actions.
    """

    observation: Observation
    action: GameAction
    reasoning: str | None


@dataclass(frozen=True)
class Context:
    """What the model acts on besides the current frame.

    The task and knowledge are long-term; memory holds the recent turns, oldest first.
    """

    task: str
    knowledge: str
    memory: tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class Turn:
    """What the agent saw and acted on in one turn, the action it chose, and why."""

    observation: Observation
    context: Context
    action: Action
    reasoning: str


@dataclass(frozen=True)
class ModelError:
    """A turn that produced no action because the model call failed."""

    observation: Observation
    context: Context
    message: str


class Agent:
    """Choose actions for one task until the model reports it complete.

    The agent touches the game only through act() and manual(); the model call in act()
    works from the observation alone, and its action is applied after it returns.
    """

    def __init__(self, model: Model, knowledge: str = "") -> None:
        self._model = model
        self._knowledge = knowledge.strip()
        self._task = ""
        self._memory: deque[MemoryEntry] = deque(maxlen=MEMORY_TURNS)

    @property
    def memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._memory)

    def set_task(self, task: str) -> None:
        """Pursue ``task``; replacing another task clears memory, whose reasoning was for it.

        Clearing the task also clears memory; operator actions taken while no task is set
        stay in memory for the next task.
        """
        task = task.strip()
        if self._task and task != self._task:
            self._memory.clear()
        self._task = task

    def clear_memory(self) -> None:
        self._memory.clear()

    def act(self, session: GameplaySession) -> tuple[Turn | ModelError, Transition | None]:
        """Observe, ask the model, and apply its action unless it is Done.

        Model failures are returned as a ModelError instead of being raised. The transition
        is None when nothing was applied.

        Raises:
            ValueError: when the agent has no task.
        """
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
        """Apply an operator's action; it enters memory like the agent's own."""
        return self._apply(session, action, None)

    def _apply(self, session: GameplaySession, action: GameAction,
               reasoning: str | None) -> Transition:
        transition = session.apply(action)
        self._memory.append(MemoryEntry(transition.before, action, reasoning))
        return transition
