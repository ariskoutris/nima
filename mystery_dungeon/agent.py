"""Goal-directed game agent and its decision-model contract."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ContextManager, Mapping, Protocol

from emulator import Frame, scene_distance

if TYPE_CHECKING:
    from mystery_dungeon.session import GameplaySession, Transition


OPTIONS = {
    "UP": "move one step up in the world, or move the menu cursor up",
    "DOWN": "move one step down in the world, or move the menu cursor down",
    "LEFT": "move one step left in the world, or move the menu cursor left",
    "RIGHT": "move one step right in the world, or move the menu cursor right",
    "A": "press the A button to interact with or talk to what the character faces; confirm a menu choice; advance visible text or dialogue; attack in a dungeon",
    "B": "press the B button to open the menu during play, cancel a command, or go back",
    "SELECT": "press SELECT to show the dungeon map",
    "WAIT": "press nothing and let the game advance",
}
COMPLETION = ("complete", "incomplete", "uncertain")
MOVEMENT = frozenset(("UP", "DOWN", "LEFT", "RIGHT"))


@dataclass(frozen=True)
class RecentStep:
    frame: Frame
    action: str
    screenshot: str


@dataclass(frozen=True)
class DecisionContext:
    goal: str
    guidance: str
    history: tuple[RecentStep, ...]
    streak: int


@dataclass(frozen=True)
class DecisionScores:
    button_scores: Mapping[str, float]
    completion_scores: Mapping[str, float]
    latency_ms: int = 0
    goal: str = ""
    guidance: str = ""
    history: tuple[RecentStep, ...] = field(default_factory=tuple)

    @property
    def choice(self) -> str:
        return max(OPTIONS, key=self.button_scores.__getitem__)

    @property
    def positive(self) -> bool:
        return self.completion_scores["complete"] > 0.5


@dataclass(frozen=True)
class Turn:
    status: str
    decision: DecisionScores | None = None
    button: str | None = None
    error: str | None = None
    needs_followup: bool = False


class DecisionModel(Protocol):
    def score(self, goal: str, guidance: str, current_frame: Frame,
              recent_steps: tuple[RecentStep, ...],
              button_options: Mapping[str, str]) -> DecisionScores: ...


def load_decision_model(spec: str) -> DecisionModel:
    module, separator, name = spec.partition(":")
    if not (module and separator and name):
        raise ValueError("decision model must be MODULE:CLASS")
    cls = getattr(importlib.import_module(module), name)
    if not isinstance(cls, type):
        raise TypeError(f"{spec} is not a class")
    instance = cls()
    if not callable(getattr(instance, "score", None)):
        raise TypeError(f"{spec} must provide score")
    return instance


def validate_scores(raw: DecisionScores) -> tuple[dict[str, float], dict[str, float]]:
    buttons = dict(raw.button_scores)
    completion = dict(raw.completion_scores)
    if buttons.keys() != OPTIONS.keys() or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in buttons.values()
    ):
        raise ValueError("model must return one finite score for every button")
    if completion.keys() != set(COMPLETION) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in completion.values()
    ):
        raise ValueError("model must return finite non-negative completion scores")
    total = sum(completion.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("completion scores must have a positive finite sum")
    return buttons, {key: value / total for key, value in completion.items()}


def is_cycle(recent: deque[tuple[Frame, str, tuple[int, int] | None]]) -> bool:
    steps = list(recent)
    for length in range(2, 7):
        if len(steps) < 3 * length or len({step[1] for step in steps[-length:]}) < 2:
            continue
        for i in range(1, 2 * length + 1):
            frame, button, position = steps[-i]
            earlier, earlier_button, earlier_position = steps[-i - length]
            if (button not in MOVEMENT or button != earlier_button or
                    (position is not None and earlier_position is not None and
                     position != earlier_position) or
                    scene_distance(frame, earlier) >= 3):
                break
        else:
            return True
    return False


def has_message_panel(frame: Frame) -> bool:
    if frame.width <= 10 or frame.height <= 145:
        return False
    offset = (145 * frame.width + 10) * 4
    red, green, blue = frame.rgba[offset:offset + 3]
    return blue > 200 and green > red


class Agent:
    """Pursue a game goal one observation and action at a time.

    A step reads the current frame, decides whether to stop or press a button,
    and records the resulting frame. GameplaySession provides emulator I/O.
    Short memory and the journal support future decisions. An operator sets
    goals today; future planning can set and prioritize them.
    """

    def __init__(self, model: DecisionModel, output: Path, model_spec: str,
                 history_frames: int = 1, completion_votes: int = 1, context: str = "",
                 run: Mapping[str, str | int] | None = None,
                 detect_cycles: bool = False, transition_wait: bool = False):
        if history_frames < 0 or completion_votes < 1:
            raise ValueError("history frames must be non-negative and completion votes positive")
        self.model = model
        self.model_spec = model_spec
        self.output = output
        self.history_frames = history_frames
        self.completion_votes = completion_votes
        self.context = context.strip()
        self.detect_cycles = detect_cycles
        self.wait_on_loop = transition_wait
        self.goal = ""
        self.steering = ""
        self.use_recent = True
        self.streak = 0
        self.completed = False
        self.history: deque[RecentStep] = deque(maxlen=history_frames or 1)
        self.events: deque[dict] = deque(maxlen=40)
        self.recent_actions: deque[tuple[Frame, str, tuple[int, int] | None]] = deque(maxlen=18)
        self.repeat_button: str | None = None
        self.repeat_count = 0
        self.pending_loop: str | None = None
        self.low_visual_change_actions = 0
        self.transition_waits = 0
        self.output.mkdir(parents=True, exist_ok=True)
        prompt = None
        prompt_bytes = getattr(model, "prompt_bytes", None)
        if prompt_bytes is not None:
            (self.output / "prompt.json").write_bytes(prompt_bytes)
            prompt = {"file": "prompt.json", "version": model.prompt_version,
                      "sha256": hashlib.sha256(prompt_bytes).hexdigest()}
        context_snapshot = None
        if self.context:
            context_bytes = self.context.encode("utf-8")
            (self.output / "context.txt").write_bytes(context_bytes)
            context_snapshot = {"file": "context.txt",
                                "sha256": hashlib.sha256(context_bytes).hexdigest()}
        (self.output / "config.json").write_text(json.dumps({
            "format_version": 2, "decision_model": model_spec,
            "history_frames": history_frames, "completion_votes": completion_votes,
            "detect_cycles": detect_cycles, "transition_wait": transition_wait,
            "button_options_sha256": hashlib.sha256(
                json.dumps(OPTIONS, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "prompt": prompt, "context": context_snapshot, "run": dict(run or {}),
        }, indent=2) + "\n")

    @property
    def guidance(self) -> str:
        return "\n\n".join(part for part in (self.context, self.steering) if part)

    def record(self, row: dict) -> None:
        row.setdefault("time", datetime.now().astimezone().isoformat(timespec="seconds"))
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "decisions.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")

    def set_goal(self, goal: str) -> bool:
        goal = goal.strip()
        if goal == self.goal:
            return False
        self.goal = goal
        self._clear_short_memory()
        self.record({"event": "goal_changed", "goal": goal})
        return True

    def set_steering(self, steering: str) -> bool:
        steering = steering.strip()
        if steering == self.steering:
            return False
        self.steering = steering
        self._clear_short_memory()
        self.record({"event": "guidance_changed", "steering": steering})
        return True

    def set_history_enabled(self, enabled: bool) -> bool:
        if enabled == self.use_recent:
            return False
        self.use_recent = enabled
        self.history.clear()
        self.streak = 0
        self._clear_loop_state()
        self.record({"event": "history_setting", "enabled": enabled})
        return True

    def _clear_loop_state(self) -> None:
        self.recent_actions.clear()
        self.repeat_button = None
        self.repeat_count = 0
        self.pending_loop = None

    def _clear_short_memory(self) -> None:
        self.history.clear()
        self.streak = 0
        self.completed = False
        self._clear_loop_state()

    def reset(self, screenshot: str) -> None:
        self._clear_short_memory()
        self.events.clear()
        self.record({"event": "reset", "screenshot": screenshot})

    def change_model(self, model: DecisionModel, spec: str) -> None:
        previous = self.model_spec
        self.model, self.model_spec = model, spec
        self._clear_short_memory()
        event = {"event": "model_change", "from": previous, "to": spec}
        if prompt_bytes := getattr(model, "prompt_bytes", None):
            digest = hashlib.sha256(prompt_bytes).hexdigest()
            path = self.output / "prompts" / f"{digest}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(prompt_bytes)
            event["prompt"] = {"file": str(path.relative_to(self.output)),
                               "version": model.prompt_version, "sha256": digest}
        self.record(event)

    def transition_wait(self, screenshot: str) -> None:
        self._clear_short_memory()
        self.record({"event": "transition_wait", "screenshot": screenshot})

    def resume(self) -> None:
        if self.completed:
            self.streak = 0
            self.completed = False
            self.record({"event": "resume", "goal": self.goal})

    def _decision_context(self) -> DecisionContext:
        return DecisionContext(self.goal, self.guidance,
                               tuple(self.history) if self.use_recent else (),
                               self.streak)

    def _decide(self, current_frame: Frame,
                context: DecisionContext) -> tuple[DecisionScores, bool]:
        started = time.perf_counter()
        raw = self.model.score(context.goal, context.guidance, current_frame,
                               context.history, OPTIONS.copy())
        buttons, completion = validate_scores(raw)
        result = DecisionScores(buttons, completion,
                                round((time.perf_counter() - started) * 1000),
                                context.goal, context.guidance, context.history)
        return result, result.positive and context.streak + 1 >= self.completion_votes

    def step(self, session: GameplaySession, *, action_lock: ContextManager | None = None,
             cancel_reason: Callable[[], str | None] | None = None,
             on_model_waiting: Callable[[bool], None] | None = None) -> Turn:
        """Make one decision and either stop or execute its action."""
        if session.current is None:
            raise RuntimeError("capture a frame before stepping")
        waited, ready = session.wait_for_text()
        if waited:
            self.record({"event": "text_wait", "frames": waited,
                         "screenshot": session.current_name, "ready": ready})
        if not ready:
            return Turn("stalled", error="dialogue text did not finish printing")
        if cancel_reason and cancel_reason():
            return Turn("cancelled")
        context = self._decision_context()
        if not context.goal:
            raise ValueError("set a goal before running")
        screenshot = session.current_name
        if on_model_waiting:
            on_model_waiting(True)
        try:
            decision, stop = self._decide(session.current, context)
        except Exception as exc:
            with action_lock or nullcontext():
                cancelled = cancel_reason() if cancel_reason else None
                self._failure(context, screenshot, exc, stale=bool(cancelled))
            return Turn("cancelled" if cancelled else "model_error",
                        error=f"{type(exc).__name__}: {exc}")
        finally:
            if on_model_waiting:
                on_model_waiting(False)
        with action_lock or nullcontext():
            cancelled = cancel_reason() if cancel_reason else None
            if cancelled:
                self._discard(decision, screenshot, cancelled)
                return Turn("cancelled", decision)
            if stop:
                self._stop(decision, screenshot)
                return Turn("completed", decision)
            if self.pending_loop:
                reason = self.pending_loop
                self._discard(decision, screenshot, reason)
                if not self.wait_on_loop:
                    self._clear_loop_state()
                    self.record({"event": "loop_stop", "reason": reason,
                                 "goal": self.goal, "screenshot": screenshot})
                    return Turn(reason, decision)
                transition = session.apply("WAIT", frames=60)
                self._observe(transition, "WAIT", "transition_wait", None)
                self.transition_wait(session.current_name)
                if scene_distance(transition.before, transition.after) >= 10:
                    self.transition_waits += 1
                    return Turn("transition", decision)
                self.record({"event": "loop_stop", "reason": reason,
                             "goal": self.goal, "screenshot": session.current_name})
                return Turn(reason, decision)
            button = decision.choice
            before_state = session.ram_state()
            transition = session.apply(button)
            after_state = session.ram_state()
            moved = (before_state.player_position != after_state.player_position
                     if before_state and after_state and before_state.player_position is not None
                     and after_state.player_position is not None else None)
            self._observe(transition, button, "agent", decision,
                          after_state.mode if after_state else None, moved,
                          after_state.player_position if after_state else None)
            return Turn("acted", decision, button, needs_followup=bool(self.pending_loop))

    def manual_step(self, session: GameplaySession, button: str) -> Frame:
        """Apply an operator action while keeping the agent's history accurate."""
        transition = session.apply(button)
        self._observe(transition, button, "manual", None)
        return transition.after

    def _observe(self, transition: Transition, button: str, source: str,
                decision: DecisionScores | None, mode: str | None = None,
                moved: bool | None = None,
                position: tuple[int, int] | None = None) -> dict:
        before, before_name = transition.before, transition.before_name
        after, after_name = transition.after, transition.after_name
        if source == "agent" and decision is not None:
            self.streak = self.streak + 1 if decision.positive else 0
        else:
            self.streak = 0
        if source == "agent":
            distance = scene_distance(before, after)
            self.low_visual_change_actions += distance < 10
            eligible = button in MOVEMENT or (button == "A" and mode != "dungeon")
            if eligible and distance < 3 and moved is not True:
                self.repeat_count = self.repeat_count + 1 if button == self.repeat_button else 1
                self.repeat_button = button
            else:
                self.repeat_count = 0
                self.repeat_button = None
            self.recent_actions.append((after, button, position))
            stalled = self.repeat_count >= 8
            cycle = self.detect_cycles and is_cycle(self.recent_actions)
            if (stalled or cycle) and not has_message_panel(after):
                self.pending_loop = "stalled" if stalled else "loop"
                self.repeat_count = 0
                self.repeat_button = None
        else:
            self._clear_loop_state()
        if self.use_recent and self.history_frames and source != "transition_wait":
            self.history.append(RecentStep(before, button, before_name))
        row = {"event": "action", "before": before_name, "after": after_name,
               "goal": self.goal, "steering": self.steering, "source": source,
               "choice": button, "scores": dict(decision.button_scores) if decision else {},
               "completion_scores": dict(decision.completion_scores) if decision else {},
               "history": [{"screenshot": step.screenshot, "action": step.action}
                           for step in decision.history] if decision else [],
               "guidance": decision.guidance if decision else self.guidance,
               "decision_model": self.model_spec, "button_options": OPTIONS if decision else {},
               "latency_ms": decision.latency_ms if decision else 0,
               "stalled": source == "agent" and self.pending_loop == "stalled",
               "index": transition.index, "step": transition.step,
               "frame_before": transition.frame_before,
               "frame_after": transition.frame_after}
        if decision is not None:
            row.update(self._model_response())
        self.record(row)
        self.events.append(row)
        if source == "agent" and self.pending_loop:
            self.record({"event": "suspected_loop", "reason": self.pending_loop,
                         "goal": self.goal, "screenshot": after_name, "button": button})
        return row

    def _stop(self, decision: DecisionScores, screenshot: str) -> None:
        self.completed = True
        self.streak = 0
        self._clear_loop_state()
        self.record({"event": "completion", **self._decision_fields(decision, screenshot)})

    def _discard(self, decision: DecisionScores, screenshot: str, reason: str) -> None:
        self.record({"event": "discarded_decision", "reason": reason,
                     **self._decision_fields(decision, screenshot)})

    def _failure(self, context: DecisionContext, screenshot: str, error: Exception,
                stale: bool = False) -> None:
        self.record({"event": "discarded_error" if stale else "model_error",
                     "error": f"{type(error).__name__}: {error}",
                     **self._request_fields(context, screenshot),
                     **self._model_response()})

    def _model_response(self) -> dict:
        response = getattr(self.model, "last_response", None)
        return {"model_response": response} if response is not None else {}

    def _request_fields(self, request: DecisionContext | DecisionScores,
                        screenshot: str) -> dict:
        return {"goal": request.goal, "screenshot": screenshot,
                "guidance": request.guidance,
                "history": [{"screenshot": step.screenshot, "action": step.action}
                            for step in request.history],
                "button_options": OPTIONS, "decision_model": self.model_spec}

    def _decision_fields(self, decision: DecisionScores, screenshot: str) -> dict:
        return {**self._request_fields(decision, screenshot),
                "scores": dict(decision.button_scores),
                "completion_scores": dict(decision.completion_scores),
                "latency_ms": decision.latency_ms,
                **self._model_response()}
