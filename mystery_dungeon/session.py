"""Emulator execution boundary shared by live play, replay, and benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from emulator import Frame, GbaEmulator
from mystery_dungeon.agent import Agent, DecisionScores, DecisionContext


class GameplaySession:
    def __init__(self, emulator: GbaEmulator, agent: Agent, output: Path,
                 hold_frames: int = 8, release_frames: int = 4):
        if min(hold_frames, release_frames) < 1:
            raise ValueError("frame counts must be positive")
        self.emulator = emulator
        self.agent = agent
        self.output = output
        self.hold_frames = hold_frames
        self.release_frames = release_frames
        self.image_index = -1
        self.frame_count = 0
        self.current: Frame | None = None
        self.current_name = ""
        self.last_row: dict | None = None

    def capture(self) -> Frame:
        """Capture the current emulator frame after a restore or initial step."""
        self.image_index += 1
        self.current = self.emulator.screenshot()
        self.current_name = f"frames/{self.image_index:06d}.png"
        (self.output / "frames").mkdir(parents=True, exist_ok=True)
        self.current.save_png(self.output / self.current_name)
        return self.current

    def decide(self, context: DecisionContext | None = None) -> tuple[DecisionScores, bool]:
        if self.current is None:
            raise RuntimeError("capture a frame before deciding")
        return self.agent.decide(self.current, context)

    def apply(self, button: str, source: str,
              decision: DecisionScores | None = None,
              frames: int | None = None,
              after_details: Callable[[Frame], dict] | None = None) -> Frame:
        if self.current is None:
            raise RuntimeError("capture a frame before applying an action")
        before, before_name, frame_before = self.current, self.current_name, self.frame_count
        if frames is None:
            self.emulator.step(self.hold_frames, () if button == "WAIT" else (button,))
            self.emulator.step(self.release_frames)
            self.frame_count += self.hold_frames + self.release_frames
        else:
            if button != "WAIT" or frames < 1:
                raise ValueError("frame override requires a positive WAIT")
            self.emulator.step(frames)
            self.frame_count += frames
        after = self.capture()
        details = {"index": self.image_index - 1, "step": self.image_index,
                   "frame_before": frame_before, "frame_after": self.frame_count}
        if after_details:
            details.update(after_details(after))
        self.last_row = self.agent.observe(
            before, before_name, button, self.current_name, source, decision,
            **details)
        return after

    def stop(self, decision: DecisionScores) -> None:
        self.agent.stop(decision, self.current_name)

    def save_latest(self) -> None:
        states = self.output / "states"
        states.mkdir(parents=True, exist_ok=True)
        temporary = states / "latest.state.tmp"
        temporary.write_bytes(self.emulator.save_state())
        temporary.replace(states / "latest.state")
