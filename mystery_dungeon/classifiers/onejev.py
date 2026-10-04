"""OneJev adapter for the screenshot classifier used by mystery_dungeon.play."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Mapping

from qev import Choice, Client


def classify(screenshot: Path, goal: str, recent: tuple[str, ...],
             options: Mapping[str, str]) -> Mapping[str, float]:
    image = base64.b64encode(screenshot.read_bytes()).decode("ascii")
    response = Client().system_one(
        state={"goal": goal, "recent_buttons": recent, "screen": "<image:1>"},
        media=[{"type": "image", "data": "data:image/png;base64," + image}],
        questions={"next": Choice("Which gamepad action should be taken next to make progress?", dict(options))},
    )
    return response.answers["next"].probabilities
