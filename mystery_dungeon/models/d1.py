"""Liquid AI d1-3B through a llama-server System One endpoint, filling both model roles."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from emulator import Frame
from mystery_dungeon.actions import ACTIONS, Action, name
from mystery_dungeon.prompts import ACT, ACTION_DESCRIPTIONS, COMPLETE

if TYPE_CHECKING:
    from mystery_dungeon.agent import MemoryEntry

TIMEOUT_SECONDS = 120


class D1Model:
    def __init__(self, url: str | None = None) -> None:
        if url is None:
            raise ValueError("d1 needs a server: set serve or url in its config table")
        self.url = url.rstrip("/") + "/v1/systemone"
        self.last_response: dict | None = None

    def choose_action(self, task: str, knowledge: str, frame: Frame,
                      memory: tuple[MemoryEntry, ...]) -> Action:
        question = f"{ACT}\n{knowledge}" if knowledge else ACT
        choice = self._ask(task, frame, memory, {
            "type": "choice", "instructions": question, "criteria": ACTION_DESCRIPTIONS}).get("choice")
        if choice not in ACTIONS:
            raise ValueError(f"d1 returned an unknown action: {choice!r}")
        return ACTIONS[choice]

    def is_complete(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...]) -> bool:
        # d1 answers a yes/no question with the probability of yes.
        return self._ask(task, frame, memory, {"type": "noul", "instructions": COMPLETE})["noul"] > 0.5

    def _ask(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...],
             question: dict) -> dict:
        self.last_response = None
        # d1 takes images as a bare list, so the state names them by position.
        frames = [entry.frame for entry in memory] + [frame]
        observations = [{"image": index, "action_after": name(entry.action)}
                        for index, entry in enumerate(memory, 1)]
        observations.append({"image": len(frames), "role": "current"})
        payload = {
            "state": json.dumps({"task": task, "observations": observations}),
            "images": [image.data_url for image in frames],
            "questions": {"answer": question},
        }
        request = Request(self.url, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                data = json.load(response)
        except HTTPError as exc:
            # llama-server explains the failure (old build, model not loaded) in the body.
            raise RuntimeError(f"d1 server returned {exc.code}: {exc.read().decode()}") from exc
        self.last_response = data
        return data["answers"]["answer"]
