"""Agent that interacts with the game to complete the given task."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mystery_dungeon.actions import Action, GameAction, Stop

if TYPE_CHECKING:
    from mystery_dungeon.model import Finding, Model, ScratchpadEdit
    from mystery_dungeon.session import GameState, Observation, Transition

MEMORY_TURNS = 5  # not yet tuned


class Session(Protocol):
    """The game as the agent sees it: GameplaySession, or live's RealtimeSession."""

    def observe(self) -> Observation: ...

    def apply(self, action: GameAction) -> Transition: ...


@dataclass(frozen=True)
class MemoryEntry:
    """An earlier turn. Reasoning is None when the operator acted."""

    observation: Observation
    action: GameAction
    reasoning: str | None


@dataclass(frozen=True)
class ScratchpadLine:
    """One line of the model's scratchpad; its number stays fixed until the task changes."""

    number: int
    text: str


@dataclass(frozen=True)
class Context:
    """Everything the model gets besides the current frame."""

    task: str
    knowledge: str
    memory: tuple[MemoryEntry, ...]
    game_state: GameState | None = None  # with the current frame
    scratchpad: tuple[ScratchpadLine, ...] = ()


@dataclass(frozen=True)
class Turn:
    observation: Observation
    context: Context
    action: Action
    reasoning: str
    scratchpad: ScratchpadEdit
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class ModelError:
    """A turn where the model call failed."""

    observation: Observation
    context: Context
    message: str


class Agent:
    """Agent that interacts with the game to complete the given task.
    Chooses actions with a model, keeps recent turns in memory and the task's scratchpad and
    findings."""

    def __init__(self, model: Model, knowledge: str = "") -> None:
        self._model = model
        self._knowledge = knowledge.strip()
        self._task = ""
        self._memory: deque[MemoryEntry] = deque(maxlen=MEMORY_TURNS)
        self._scratchpad: list[ScratchpadLine] = []
        self._next_line = 1
        self._findings: list[Finding] = []

    @property
    def memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._memory)

    @property
    def scratchpad(self) -> tuple[ScratchpadLine, ...]:
        return tuple(self._scratchpad)

    @property
    def findings(self) -> tuple[Finding, ...]:
        """The task's findings so far; they are logged, never shown to the model."""
        return tuple(self._findings)

    def set_task(self, task: str) -> None:
        """Switch tasks, forgetting the old task's memory, scratchpad and findings."""
        task = task.strip()
        if self._task and task != self._task:
            self.clear_memory()
        self._task = task

    def clear_memory(self) -> None:
        self._memory.clear()
        self._scratchpad.clear()
        self._next_line = 1
        self._findings.clear()

    def act(self, session: Session) -> tuple[Turn | ModelError, Transition | None]:
        """Take one turn: observe, ask the model and apply its action and scratchpad edit."""
        if not self._task:
            raise ValueError("give the agent a task before it acts")
        observation = session.observe()
        context = Context(self._task, self._knowledge, self.memory, observation.game_state,
                          self.scratchpad)
        try:
            reply = self._model.call(context, observation.frame)
            unknown = set(reply.scratchpad.remove) - {line.number for line in self._scratchpad}
            if unknown:
                raise ValueError(f"removed scratchpad lines that do not exist: {sorted(unknown)}")
        except Exception as exc:
            return ModelError(observation, context, f"{type(exc).__name__}: {exc}"), None
        turn = Turn(observation, context, reply.action, reply.reasoning, reply.scratchpad,
                    reply.findings)
        match reply.action:
            case Stop():
                transition = None
            case action:
                transition = self._apply(session, action, reply.reasoning)
        self._edit(reply.scratchpad)
        self._findings.extend(reply.findings)
        return turn, transition

    def _edit(self, edit: ScratchpadEdit) -> None:
        """Remove lines first, so an edit is a removal plus an added line."""
        self._scratchpad = [line for line in self._scratchpad if line.number not in edit.remove]
        for text in edit.add:
            self._scratchpad.append(ScratchpadLine(self._next_line, text))
            self._next_line += 1

    def manual(self, session: Session, action: GameAction) -> Transition:
        """Apply an operator's action."""
        return self._apply(session, action, None)

    def _apply(self, session: Session, action: GameAction,
               reasoning: str | None) -> Transition:
        transition = session.apply(action)
        self._memory.append(MemoryEntry(transition.before, action, reasoning))
        return transition
