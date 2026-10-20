"""Record a run's configuration and events."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from mystery_dungeon.actions import Done, GameAction, to_json

if TYPE_CHECKING:
    from emulator import Frame
    from mystery_dungeon.agent import Context, ModelError, Turn
    from mystery_dungeon.model import Model, Reply
    from mystery_dungeon.session import Observation, Transition

# Shared by run and benchmark configs. 9: no discarded rows; operator input waits for the
# turn in progress instead of cancelling it.
FORMAT_VERSION = 9
RECENT_ROWS = 40


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
    """Write config.json once, then append one events.jsonl row per event.

    Calls made through ``model`` are attached to the next recorded turn.
    """

    def __init__(self, output: Path, model: Model, run: Mapping[str, str | int],
                 knowledge: str = "") -> None:
        output.mkdir(parents=True, exist_ok=True)
        self._output = output
        self._calls: list[dict] = []
        self.model = LoggedModel(model, self._calls)
        self._recent: deque[dict] = deque(maxlen=RECENT_ROWS)
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
        """Return action rows since the last reset, newest first, without responses or memory."""
        return list(reversed(self._recent))

    def record_turn(self, turn: Turn, transition: Transition | None) -> None:
        """Record an agent turn; transition is what its game action did, None for Done.

        Raises:
            ValueError: when a game action comes without its transition.
        """
        fields = {**self._step_fields(turn.observation, turn.context),
                  "reasoning": turn.reasoning}
        match turn.action:
            case Done():
                self._write("completion", fields)
            case action if transition is not None:
                row = self._write("action", {**fields, "action": to_json(action),
                                             **_transition_fields(transition)})
                self._recent.append(_summary(row))
            case _:
                raise ValueError("a game action needs the transition it caused")

    def record_model_error(self, error: ModelError) -> None:
        self._write("model_error", {**self._step_fields(error.observation, error.context),
                                    "error": error.message})

    def record_manual(self, action: GameAction, transition: Transition) -> None:
        self._recent.append(_summary(self._write("action", {
            "source": "manual", "action": to_json(action),
            **_transition_fields(transition)})))

    def record_reset(self, screenshot: str) -> None:
        self._recent.clear()
        self._write("reset", {"screenshot": screenshot})

    def record_run_stop(self, reason: str, screenshot: str) -> None:
        self._write("run_stop", {"reason": reason, "screenshot": screenshot})

    def _step_fields(self, observation: Observation, context: Context) -> dict:
        """Describe what the agent acted on and take the model calls made since the last row."""
        calls, self._calls[:] = self._calls.copy(), []
        return {"source": "agent", "task": context.task, "screenshot": observation.screenshot,
                "memory": [{"screenshot": entry.observation.screenshot,
                            "action": to_json(entry.action), "reasoning": entry.reasoning}
                           for entry in context.memory],
                "model_calls": calls}

    def _write(self, event: str, fields: dict) -> dict:
        row = {"event": event, "time": datetime.now().astimezone().isoformat(timespec="seconds"),
               **fields}
        with (self._output / "events.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        return row


def _transition_fields(transition: Transition) -> dict:
    return {"screenshot": transition.before.screenshot, "after": transition.after.screenshot,
            "index": transition.index, "action_number": transition.action_number,
            "frame_before": transition.frame_before, "frame_after": transition.frame_after}


def _summary(row: dict) -> dict:
    """An action row as the live view lists it: model responses and memory replaced by total ms."""
    summary = {key: value for key, value in row.items() if key not in ("model_calls", "memory")}
    summary["model_ms"] = sum(call["ms"] for call in row.get("model_calls", ()))
    return summary
