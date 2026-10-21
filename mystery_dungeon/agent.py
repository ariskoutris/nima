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
EDIT_REJECTED = ("A scratchpad edit of yours was rejected, so none of it applied: {problem}. "
                 "Delete this line once you have redone the edit.")


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
class Context:
    """Everything the model gets besides the current frame."""

    task: str
    knowledge: str
    memory: tuple[MemoryEntry, ...]
    game_state: GameState  # with the current frame
    scratchpad: tuple[str, ...] = ()  # shown numbered from 1


@dataclass(frozen=True)
class Turn:
    observation: Observation
    context: Context
    action: Action
    reasoning: str
    scratchpad: ScratchpadEdit
    findings: tuple[Finding, ...]
    scratchpad_problem: str | None  # why the edit was rejected; None when it applied


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
        self._scratchpad: list[str] = []
        self._findings: list[Finding] = []

    @property
    def memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._memory)

    @property
    def scratchpad(self) -> tuple[str, ...]:
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
        except Exception as exc:
            return ModelError(observation, context, f"{type(exc).__name__}: {exc}"), None
        # A bad edit costs only itself: in a live run one discarded a correct STOP.
        problem = reply.scratchpad.problem(len(self._scratchpad))
        turn = Turn(observation, context, reply.action, reply.reasoning, reply.scratchpad,
                    reply.findings, problem)
        match reply.action:
            case Stop():
                transition = None
            case action:
                transition = self._apply(session, action, reply.reasoning)
        if problem is None:
            self._edit(reply.scratchpad)
        else:
            self._scratchpad.append(EDIT_REJECTED.format(problem=problem))
        self._findings.extend(reply.findings)
        return turn, transition

    def _edit(self, edit: ScratchpadEdit) -> None:
        """Apply a valid edit by the numbers the model saw."""
        if edit.rewrite is not None:
            self._scratchpad = list(edit.rewrite)
            return
        changes = {change.line: change.text for change in edit.lines}
        self._scratchpad = [text for number, line in enumerate(self._scratchpad, 1)
                            if (text := changes.get(number, line)) is not None]
        self._scratchpad.extend(edit.add)

    def manual(self, session: Session, action: GameAction) -> Transition:
        """Apply an operator's action."""
        return self._apply(session, action, None)

    def _apply(self, session: Session, action: GameAction,
               reasoning: str | None) -> Transition:
        transition = session.apply(action)
        self._memory.append(MemoryEntry(transition.before, action, reasoning))
        return transition
