"""Agent that interacts with the game to complete the given task."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mystery_dungeon.actions import Action, GameAction, Stop

if TYPE_CHECKING:
    from mystery_dungeon.model import Model
    from mystery_dungeon.session import GameState, Observation, Transition

MEMORY_TURNS = 5  # not yet tuned


class Session(Protocol):
    """The game as the agent sees it: GameplaySession, or RealtimeSession around one."""

    def observe(self) -> Observation: ...

    def apply(self, action: GameAction) -> Transition: ...


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
    game_state: GameState  # with the current frame
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Turn:
    observation: Observation
    context: Context
    action: Action
    reasoning: str
    note: str | None


@dataclass(frozen=True)
class ModelError:
    """A turn where the model call failed."""

    observation: Observation
    context: Context
    message: str


class Agent:
    """Agent that interacts with the game to complete the given task.
    Chooses actions with a model and keeps short-term memory, plus the task's turns for its
    review."""

    def __init__(self, model: Model, knowledge: str = "") -> None:
        self._model = model
        self._knowledge = knowledge.strip()
        self._task = ""
        self._memory: deque[MemoryEntry] = deque(maxlen=MEMORY_TURNS)
        self._notes: list[str] = []
        self._turns: list[Turn] = []

    @property
    def memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._memory)

    @property
    def notes(self) -> tuple[str, ...]:
        """The task's notes, oldest first; a note is never changed or removed."""
        return tuple(self._notes)

    @property
    def turns(self) -> tuple[Turn, ...]:
        """The task's turns since the last clear_turns, oldest first, for its review."""
        return tuple(self._turns)

    def clear_turns(self) -> None:
        """Forget the turns kept for the review, once they are reviewed."""
        self._turns.clear()

    def set_task(self, task: str) -> None:
        """Switch tasks, forgetting the old task's memory, notes and turns."""
        task = task.strip()
        if self._task and task != self._task:
            self.clear_memory()
        self._task = task

    def clear_memory(self) -> None:
        self._memory.clear()
        self._notes.clear()
        self._turns.clear()

    def act(self, session: Session) -> tuple[Turn | ModelError, Transition | None]:
        """Take one turn: observe, ask the model and apply its action and note."""
        if not self._task:
            raise ValueError("give the agent a task before it acts")
        observation = session.observe()
        context = Context(self._task, self._knowledge, self.memory, observation.game_state,
                          self.notes)
        try:
            reply = self._model.call(context, observation.frame)
        except Exception as exc:
            return ModelError(observation, context, f"{type(exc).__name__}: {exc}"), None
        turn = Turn(observation, context, reply.action, reply.reasoning, reply.note)
        match reply.action:
            case Stop():
                transition = None
            case action:
                transition = self._apply(session, action, reply.reasoning)
        # The model repeats a note when it forgets it already wrote it, sometimes changing only
        # the case or the final period (seen in the guided notes runs).
        if reply.note and _same(reply.note) not in map(_same, self._notes):
            self._notes.append(reply.note)
        self._turns.append(turn)
        return turn, transition

    def manual(self, session: Session, action: GameAction) -> Transition:
        """Apply an operator's action."""
        return self._apply(session, action, None)

    def _apply(self, session: Session, action: GameAction,
               reasoning: str | None) -> Transition:
        transition = session.apply(action)
        self._memory.append(MemoryEntry(transition.before, action, reasoning))
        return transition


def _same(note: str) -> str:
    """A note's text for spotting repeats: case and final punctuation ignored."""
    return note.casefold().rstrip(" .!")
