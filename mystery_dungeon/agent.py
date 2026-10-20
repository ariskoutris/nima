"""Pursue one task: act on the current frame and recent turns until the model reports DONE."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from emulator import Frame
from mystery_dungeon.actions import Action

if TYPE_CHECKING:
    from mystery_dungeon.model import Model
    from mystery_dungeon.session import GameplaySession, Observation, Transition

MEMORY_TURNS = 5  # earlier turns every model call sees; not yet tuned


@dataclass(frozen=True)
class MemoryEntry:
    """A screenshot, the action taken right after it, and the reasoning behind that action.

    Reasoning is None for operator actions.
    """

    frame: Frame
    screenshot: str
    action: Action
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
    """What the agent proposed for one observation; nothing has been applied yet.

    A turn without an action or error reports the task complete.
    """

    observation: Observation
    context: Context
    action: Action | None = None
    reasoning: str | None = None
    error: str | None = None

    @property
    def status(self) -> Literal["model_error", "completion", "action"]:
        if self.error is not None:
            return "model_error"
        return "completion" if self.action is None else "action"


class Agent:
    """Choose actions for one task until the model reports it complete.

    The agent touches the game only through commit() and manual(); everything
    else works from observations.
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

    def propose(self, observation: Observation) -> Turn:
        """Choose an action or report completion, without touching the game.

        Model failures are returned as the turn's error instead of being raised.

        Raises:
            ValueError: when the agent has no task.
        """
        if not self._task:
            raise ValueError("give the agent a task before it acts")
        context = Context(self._task, self._knowledge, self.memory)
        try:
            choice = self._model.choose(context, observation.frame)
        except Exception as exc:
            return Turn(observation, context, error=f"{type(exc).__name__}: {exc}")
        return Turn(observation, context, choice.action, choice.reasoning)

    def commit(self, session: GameplaySession, turn: Turn) -> Transition | None:
        """Apply a proposed action and remember it; turns without an action change nothing."""
        if turn.action is None:
            return None
        return self._apply(session, turn.action, turn.reasoning)

    def manual(self, session: GameplaySession, action: Action) -> Transition:
        """Apply an operator's action; it enters memory like the agent's own."""
        return self._apply(session, action, None)

    def take_turn(self, session: GameplaySession) -> tuple[Turn, Transition | None]:
        """Observe, propose and commit in one call for single-threaded runners."""
        turn = self.propose(session.observe())
        return turn, self.commit(session, turn)

    def _apply(self, session: GameplaySession, action: Action,
               reasoning: str | None) -> Transition:
        transition = session.apply(action)
        self._memory.append(MemoryEntry(transition.before, transition.before_name, action,
                                        reasoning))
        return transition
