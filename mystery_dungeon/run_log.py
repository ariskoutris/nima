"""Record a run's configuration and events."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from mystery_dungeon.actions import Action, to_json

if TYPE_CHECKING:
    from emulator import Frame
    from mystery_dungeon.agent import Context, Turn
    from mystery_dungeon.model import Choice, Model
    from mystery_dungeon.session import Transition

# Shared by run and benchmark configs. 8: agent rows carry the model's reasoning, one call
# per turn claims completion, rows drop task_changed and the per-row model name.
FORMAT_VERSION = 8
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

    def choose(self, context: Context, frame: Frame) -> Choice:
        started = time.perf_counter()
        try:
            return self._model.choose(context, frame)
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
        """Record a turn the agent committed; transition is None when nothing was applied."""
        fields = self._turn_fields(turn)
        if transition is None:
            self._write(turn.status, fields)
            return
        row = self._write(turn.status, {**fields, **_transition_fields(transition)})
        self._recent.append(_summary(row))

    def record_discarded(self, turn: Turn, reason: str) -> None:
        """Record a turn that was proposed but never applied."""
        self._write("discarded", {"reason": reason, **self._turn_fields(turn)})

    def record_manual(self, action: Action, transition: Transition) -> None:
        self._recent.append(_summary(self._write("action", {
            "source": "manual", "action": to_json(action),
            **_transition_fields(transition)})))

    def record_reset(self, screenshot: str) -> None:
        self._recent.clear()
        self._write("reset", {"screenshot": screenshot})

    def record_run_stop(self, reason: str, screenshot: str) -> None:
        self._write("run_stop", {"reason": reason, "screenshot": screenshot})

    def _turn_fields(self, turn: Turn) -> dict:
        """Describe a turn and take the model calls made since the previous turn."""
        calls, self._calls[:] = self._calls.copy(), []
        fields = {"source": "agent", "task": turn.context.task,
                  "screenshot": turn.observation.screenshot,
                  "memory": [{"screenshot": entry.screenshot, "action": to_json(entry.action),
                              "reasoning": entry.reasoning} for entry in turn.context.memory],
                  "model_calls": calls}
        if turn.action is not None:
            fields["action"] = to_json(turn.action)
        if turn.reasoning is not None:
            fields["reasoning"] = turn.reasoning
        if turn.error is not None:
            fields["error"] = turn.error
        return fields

    def _write(self, event: str, fields: dict) -> dict:
        row = {"event": event, "time": datetime.now().astimezone().isoformat(timespec="seconds"),
               **fields}
        with (self._output / "events.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        return row


def _transition_fields(transition: Transition) -> dict:
    return {"screenshot": transition.before_name, "after": transition.after_name,
            "index": transition.index, "action_number": transition.action_number,
            "frame_before": transition.frame_before, "frame_after": transition.frame_after}


def _summary(row: dict) -> dict:
    """An action row as the live view lists it: model responses and memory replaced by total ms."""
    summary = {key: value for key, value in row.items() if key not in ("model_calls", "memory")}
    summary["model_ms"] = sum(call["ms"] for call in row.get("model_calls", ()))
    return summary
