"""Emulator execution boundary shared by live play, replay, and benchmarks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from emulator import Frame, GbaEmulator, scene_distance
from mystery_dungeon.ram_state import ROM_SHA1, RamState, read_ram_state


@dataclass(frozen=True)
class Transition:
    before: Frame
    before_name: str
    after: Frame
    after_name: str
    index: int
    step: int
    frame_before: int
    frame_after: int


def _visible_cursor_frame(first: Frame, second: Frame) -> Frame:
    """Pick the brighter of the two frames at the strongest local blink."""
    if (first.width, first.height) != (second.width, second.height):
        return second
    blocks: dict[tuple[int, int], int] = {}
    for index in range(0, len(first.rgba), 4):
        before = first.rgba[index:index + 3]
        after = second.rgba[index:index + 3]
        if sum(abs(a - b) for a, b in zip(before, after)) <= 100:
            continue
        pixel = index // 4
        block = (pixel % first.width // 16, pixel // first.width // 16)
        blocks[block] = blocks.get(block, 0) + sum(after) - sum(before)
    # ponytail: local brightness is a visual heuristic; use cursor detection if another menu animation dominates it.
    strongest = max(blocks.values(), key=abs, default=0)
    return second if strongest >= 0 else first


class GameplaySession:
    def __init__(self, emulator: GbaEmulator, output: Path,
                 hold_frames: int = 8, release_frames: int = 4):
        if min(hold_frames, release_frames) < 1:
            raise ValueError("frame counts must be positive")
        self.emulator = emulator
        self.output = output
        self.hold_frames = hold_frames
        self.release_frames = release_frames
        self.image_index = -1
        self.action_count = 0
        self.frame_count = 0
        self.current: Frame | None = None
        self.current_name = ""
        self.warning: str | None = None
        self._last_button: str | None = None
        self._unchanged_count = 0
        self.ram_enabled = (isinstance(emulator, GbaEmulator) and
                            hashlib.sha1(emulator.rom.read_bytes()).hexdigest() == ROM_SHA1)

    def ram_state(self) -> RamState | None:
        return read_ram_state(self.emulator) if self.ram_enabled else None

    def clear_warning(self) -> None:
        self.warning = None
        self._last_button = None
        self._unchanged_count = 0

    def wait_for_text(self, max_frames: int = 300) -> tuple[int, bool]:
        """Advance silent frames until the current dialogue page is fully printed."""
        if self.current is None:
            raise RuntimeError("capture a frame before waiting")
        state = self.ram_state()
        waited = 0
        while state is not None and state.text_printing and waited < max_frames:
            frames = min(4, max_frames - waited)
            self.emulator.step(frames)
            self.frame_count += frames
            waited += frames
            state = self.ram_state()
        if waited:
            if state is not None and not state.text_printing and state.mode == "menu":
                # The shop opens its choice menu a few frames after the last glyph.
                self.emulator.step(12)
                self.frame_count += 12
                waited += 12
            self.capture()
        return waited, state is None or not state.text_printing

    def capture(self, frame: Frame | None = None) -> Frame:
        """Save the current or supplied frame as the session observation."""
        self.image_index += 1
        self.current = frame if frame is not None else self.emulator.screenshot()
        self.current_name = f"frames/{self.image_index:06d}.png"
        (self.output / "frames").mkdir(parents=True, exist_ok=True)
        self.current.save_png(self.output / self.current_name)
        return self.current

    def current_frame(self) -> Frame:
        """Return one observation, waiting through a cursor blink in menus."""
        if self.current is None:
            raise RuntimeError("capture a frame before observing")
        state = self.ram_state()
        if state is None or state.mode != "menu":
            return self.current
        first = self.current
        self.emulator.step(8)
        self.frame_count += 8
        second = self.emulator.screenshot()
        if (next_state := self.ram_state()) is None or next_state.mode != "menu":
            return self.capture(second)
        return self.capture(_visible_cursor_frame(first, second))

    def observe(self) -> Frame:
        """Return a readable frame without skipping cutscenes or advancing dialogue."""
        self.wait_for_text()
        return self.current_frame()

    def apply(self, button: str, frames: int | None = None) -> Transition:
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
        self.action_count += 1
        if button != "WAIT" and scene_distance(before, after) < 3:
            self._unchanged_count = self._unchanged_count + 1 if button == self._last_button else 1
        else:
            self._unchanged_count = 0
        self._last_button = button
        self.warning = (f"Possible stall: little screen change after {self._unchanged_count} {button} presses"
                        if self._unchanged_count >= 8 else None)
        return Transition(before, before_name, after, self.current_name,
                          self.image_index - 1, self.action_count,
                          frame_before, self.frame_count)

    def save_latest(self) -> None:
        states = self.output / "states"
        states.mkdir(parents=True, exist_ok=True)
        temporary = states / "latest.state.tmp"
        temporary.write_bytes(self.emulator.save_state())
        temporary.replace(states / "latest.state")
