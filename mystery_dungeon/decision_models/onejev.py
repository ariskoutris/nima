"""OneJev decision model for buttons and visible goal completion."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Mapping

from qev import Choice, Client

from emulator import Frame
from mystery_dungeon.agent import COMPLETION, DecisionScores, RecentStep


class OneJevDecisionModel:
    def __init__(self):
        self.prompt_version = os.getenv("QEV_PROMPT_VERSION", "v1")
        if not re.fullmatch(r"v[1-9][0-9]*", self.prompt_version):
            raise ValueError("QEV_PROMPT_VERSION must be a version such as v1")
        path = Path(__file__).parents[1] / "prompts" / f"onejev-{self.prompt_version}.json"
        self.prompt_bytes = path.read_bytes()
        self.prompt = json.loads(self.prompt_bytes)
        if not isinstance(self.prompt, dict):
            raise ValueError(f"invalid OneJev prompt bundle: {path}")
        completion_options = self.prompt.get("completion_options")
        if (self.prompt.get("version") != self.prompt_version or
                not isinstance(completion_options, dict) or
                completion_options.keys() != set(COMPLETION) or
                not all(isinstance(value, str) and value for value in completion_options.values()) or
                not all(isinstance(self.prompt.get(key), str)
                        for key in ("action_question", "completion_question"))):
            raise ValueError(f"invalid OneJev prompt bundle: {path}")
        self.client = Client()

    def score(self, goal: str, guidance: str, current_frame: Frame,
              recent_steps: tuple[RecentStep, ...],
              button_options: Mapping[str, str]) -> DecisionScores:
        frames = [step.frame for step in recent_steps] + [current_frame]
        media = [{"type": "image", "data": "data:image/png;base64," +
                  base64.b64encode(frame.png_bytes()).decode("ascii")}
                 for frame in frames]
        observations = [
            {"frame": f"<image:{index}>", "action_after": step.action}
            for index, step in enumerate(recent_steps, 1)
        ]
        observations.append({"frame": f"<image:{len(frames)}>", "role": "current"})
        action_question = self.prompt["action_question"]
        if guidance:
            action_question += "\n" + guidance
        response = self.client.system_one(
            state={"observations": observations, "goal": goal},
            media=media,
            questions={
                "action": Choice(action_question, dict(button_options)),
                "completion": Choice(self.prompt["completion_question"],
                                     self.prompt["completion_options"]),
            },
        )
        return DecisionScores(response.answers["action"].probabilities,
                              response.answers["completion"].probabilities)
