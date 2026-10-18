"""OneJev through a qev System One server, filling both model roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qev import Choice, Client, Noul

from emulator import Frame
from mystery_dungeon.actions import ACTIONS, Action, name
from mystery_dungeon.prompts import ACT, ACTION_DESCRIPTIONS, COMPLETE

if TYPE_CHECKING:
    from mystery_dungeon.agent import MemoryEntry


class OneJevModel:
    def __init__(self, url: str | None = None) -> None:
        # Without a URL qev's Client would fall back to environment variables.
        if url is None:
            raise ValueError("OneJev needs a server: set serve or url in its config table")
        self.client = Client(url)
        self.last_response: dict | None = None

    def choose_action(self, task: str, knowledge: str, frame: Frame,
                      memory: tuple[MemoryEntry, ...]) -> Action:
        question = f"{ACT}\n{knowledge}" if knowledge else ACT
        answer = self._ask(task, frame, memory, Choice(question, ACTION_DESCRIPTIONS))
        if answer.choice not in ACTIONS:
            raise ValueError(f"OneJev returned an unknown action: {answer.choice!r}")
        return ACTIONS[answer.choice]

    def is_complete(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...]) -> bool:
        # qev answers a yes/no question with the probability of yes.
        return self._ask(task, frame, memory, Noul(COMPLETE)).noul > 0.5

    def _ask(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...],
             question: Choice | Noul):
        self.last_response = None
        frames = [entry.frame for entry in memory] + [frame]
        media = [{"type": "image", "data": image.data_url} for image in frames]
        observations = [{"frame": f"<image:{index}>", "action_after": name(entry.action)}
                        for index, entry in enumerate(memory, 1)]
        observations.append({"frame": f"<image:{len(frames)}>", "role": "current"})
        response = self.client.system_one(
            state={"observations": observations, "task": task},
            media=media, questions={"answer": question})
        self.last_response = {"answer": response.answers["answer"].raw, "usage": response.usage}
        return response.answers["answer"]
