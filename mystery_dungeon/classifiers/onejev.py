"""OneJev adapter for the screenshot classifier used by mystery_dungeon.play."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Mapping

from qev import Choice, Client


def inspect(screenshot: Path, question: str,
            options: Mapping[str, str]) -> Mapping[str, float]:
    return _score(screenshot, {"screen": "<image:1>"}, question, options)


def _score(screenshot: Path, state: dict, question: str,
           options: Mapping[str, str]) -> Mapping[str, float]:
    image = base64.b64encode(screenshot.read_bytes()).decode("ascii")
    response = Client().system_one(
        state=state,
        media=[{"type": "image", "data": "data:image/png;base64," + image}],
        questions={"next": Choice(question, dict(options))},
    )
    return response.answers["next"].probabilities


def classify(screenshot: Path, goal: str, recent: tuple[str, ...],
             options: Mapping[str, str]) -> Mapping[str, float]:
    # Repeated button counts reinforced blocked choices in saved-screen replays.
    return _score(
        screenshot,
        {"screen": "<image:1>", "task": {"instruction": goal}},
        "Which gamepad action should be taken next to make progress?",
        options,
    )
