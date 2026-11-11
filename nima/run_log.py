"""Record a run's configuration, turns and model calls."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

from nima.actions import GameAction, Stop, to_json
from nima.model import internal_reasoning, token_usage
from nima.review import Finding, ReviewError

if TYPE_CHECKING:
    from emulator import Frame
    from nima.actions import Outcome
    from nima.agent import Context, ModelError, Turn
    from nima.model import Model, Reply
    from nima.session import Observation, Transition

FORMAT_VERSION = 22  # of run and benchmark output; bump when a field changes
RECENT_ROWS = 50  # turn messages retained in the live feed

Ending = Literal["stop", "limit", "model_error"]  # how an attempt ended


class LoggedModel:
    """Delegate to a model while timing each call and keeping its raw response for the log."""

    def __init__(self, model: Model, calls: list[dict]) -> None:
        self._model = model
        self._calls = calls
        self.name = model.name

    @property
    def last_response(self) -> dict | None:
        return self._model.last_response

    def call(self, context: Context, frame: Frame) -> Reply:
        started = time.perf_counter()
        try:
            return self._model.call(context, frame)
        finally:
            self._calls.append({"ms": round((time.perf_counter() - started) * 1000),
                                "response": self._model.last_response})


class RunLog:
    """Write config.json once, then append rows to turns.jsonl and model_calls.jsonl.

    turns.jsonl holds what happened in the game; model_calls.jsonl holds what the model
    saw and answered, linked to its turn row by ``turn``.
    """

    def __init__(self, output: Path, model: Model, run: Mapping[str, str | int],
                 knowledge: str = "") -> None:
        output.mkdir(parents=True, exist_ok=True)
        self._output = output
        self._calls: list[dict] = []
        self.model = LoggedModel(model, self._calls)
        self._turns = 0
        self._recent: deque[dict] = deque(maxlen=RECENT_ROWS)
        self._recent_id = 0
        snapshot = None
        if knowledge:
            data = knowledge.encode("utf-8")
            (output / "knowledge.txt").write_bytes(data)
            snapshot = {"file": "knowledge.txt", "sha256": hashlib.sha256(data).hexdigest()}
        (output / "config.json").write_text(json.dumps({
            "format_version": FORMAT_VERSION, "model": model.name,
            "knowledge": snapshot, "run": dict(run),
        }, indent=2) + "\n")

    def recent(self) -> list[dict]:
        """Return turns and manual actions since the last reset, newest first, with feed IDs."""
        return list(reversed(self._recent))

    def record_turn(self, turn: Turn, transition: Transition | None) -> None:
        """Record an agent turn; transition is what its game action did, None for Stop.

        Raises:
            ValueError: when a game action comes without its transition.
        """
        match turn.action:
            case Stop(outcome):
                row = self._write_turn("stop", turn.observation, turn.context, {
                    **_reply_fields(turn), "outcome": outcome})
                self._remember(row)
            case action if transition is not None:
                row = self._write_turn("action", turn.observation, turn.context, {
                    **_reply_fields(turn), "action": to_json(action),
                    **_transition_fields(transition)})
                self._remember(row)
            case _:
                raise ValueError("a game action needs the transition it caused")

    def record_model_error(self, error: ModelError) -> None:
        self._write_turn("model_error", error.observation, error.context,
                         {"error": error.message})

    def record_manual(self, action: GameAction, transition: Transition) -> None:
        self._remember(self._write("action", {
            "source": "manual", "action": to_json(action), "model_ms": 0,
            **_transition_fields(transition)}))

    def record_reset(self, screenshot: str) -> None:
        self._recent.clear()
        self._recent_id = 0
        self._write("reset", {"screenshot": screenshot})

    def record_run_stop(self, ended: Ending, outcome: Outcome | None, actions: int, limit: int,
                        screenshot: str) -> None:
        """Record how the attempt ended; outcome is the Stop's, None for any other ending."""
        self._write("run_stop", {"ended": ended, "outcome": outcome, "actions": actions,
                                 "limit": limit, "screenshot": screenshot})

    def record_review(self, task: str, result: list[Finding] | ReviewError) -> None:
        """Record the review of a task's turns: its findings, or why it failed."""
        match result:
            case ReviewError(message):
                self._write("review", {"task": task, "findings": None, "error": message})
            case findings:
                self._write("review", {"task": task, "error": None,
                                       "findings": [dataclasses.asdict(finding)
                                                    for finding in findings]})

    def _remember(self, row: dict) -> None:
        self._recent_id += 1
        self._recent.append({**row, "id": self._recent_id})

    def _write_turn(self, event: str, observation: Observation, context: Context,
                    fields: dict) -> dict:
        """Write the model calls made since the last turn, then the turn row."""
        self._turns += 1
        calls, self._calls[:] = self._calls.copy(), []
        model_input = {"task": context.task,
                       "notes": list(context.notes),
                       "screenshot": observation.screenshot,
                       "game_state": dataclasses.asdict(context.game_state),
                       "memory": [{"screenshot": entry.observation.screenshot,
                                   "game_state": dataclasses.asdict(entry.observation.game_state),
                                   "action": to_json(entry.action),
                                   "reasoning": entry.reasoning}
                                  for entry in context.memory]}
        for call in calls:
            _append(self._output / "model_calls.jsonl", {
                "turn": self._turns, "ms": call["ms"], "tokens": token_usage(call["response"]),
                "input": model_input,
                "internal_reasoning": internal_reasoning(call["response"]),
                "response": call["response"]})
        return self._write(event, {"source": "agent", "turn": self._turns,
                                   "task": context.task, "screenshot": observation.screenshot,
                                   "model_ms": sum(call["ms"] for call in calls), **fields})

    def _write(self, event: str, fields: dict) -> dict:
        return _append(self._output / "turns.jsonl", {
            "event": event, "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            **fields})


def _reply_fields(turn: Turn) -> dict:
    return {"reasoning": turn.reasoning, "note": turn.note}


def _append(path: Path, row: dict) -> dict:
    with path.open("a") as log:
        log.write(json.dumps(row) + "\n")
    return row


def _transition_fields(transition: Transition) -> dict:
    return {"screenshot": transition.before.screenshot, "after": transition.after.screenshot,
            "index": transition.index, "action_number": transition.action_number,
            "tick_before": transition.tick_before, "tick_after": transition.tick_after}
