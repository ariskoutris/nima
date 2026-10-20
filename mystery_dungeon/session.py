"""Emulator I/O: load checkpoints, observe frames, and apply actions."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from emulator import Frame, GbaEmulator, scene_distance
from mystery_dungeon.actions import GameAction, Press, Wait, name
from mystery_dungeon.ram_state import GameRam, RamState

HOLD_TICKS = 8  # ticks each action holds its input
RELEASE_TICKS = 4  # ticks with no input after each action
TEXT_WAIT_LIMIT = 300  # most ticks spent waiting for one page of text to print
TEXT_POLL_TICKS = 4  # ticks between text-printing checks
# Menus that follow dialogue (e.g. the shop's choices) open a few ticks after the last glyph.
MENU_AFTER_TEXT_TICKS = 12
# Ticks between the two menu captures, so one of them shows the blinking cursor;
# test_menu_observation_shows_blinking_cursor_without_an_action covers it.
CURSOR_BLINK_TICKS = 8
BLINK_PIXEL_DELTA = 100  # summed RGB difference for a pixel to count as blinking
BLINK_BLOCK_PIXELS = 16  # side of the square regions compared for the strongest blink
STALL_SCENE_DISTANCE = 3  # scene_distance below which a press barely changed the screen
STALL_PRESSES = 8  # repeated barely-changing presses before warning the operator
# Characters within this many pixels of the player across and down count as on screen: the
# screen is 240x160 and the camera centres the player except at map edges.
VISIBLE_X, VISIBLE_Y = 120, 80


@dataclass(frozen=True)
class Checkpoint:
    """A core state, with the battery RAM it expects when the save state carries one."""

    state: bytes
    save_ram: bytes | None = None


@dataclass(frozen=True)
class Character:
    """Another character on screen, positioned like the player."""

    name: str
    position: tuple[int, int]
    direction: str


@dataclass(frozen=True)
class GameState:
    """What the model may know from RAM besides the frame. Positions are (x, y), y downward."""

    position: tuple[int, int] | None  # pixels, on ground maps
    tile: tuple[int, int] | None  # in dungeons
    direction: str | None
    characters: tuple[Character, ...] = ()  # on screen, on ground maps


@dataclass(frozen=True)
class Observation:
    """A frame, the path of its screenshot relative to the session output, and the game state
    when it was captured (None without RAM)."""

    frame: Frame
    screenshot: str
    game_state: GameState | None = None


@dataclass(frozen=True)
class Transition:
    """What one action did.

    ``index`` is the image index of ``before``, ``action_number`` counts actions since the
    last load, and the ticks are the session's tick count when the action started and ended.
    """

    before: Observation
    after: Observation
    index: int
    action_number: int
    tick_before: int
    tick_after: int


def read_ss0(path: Path | bytes) -> Checkpoint:
    """Extract the mGBA core state and battery RAM from its PNG save state.

    Raises:
        ValueError: when the data is not a complete, intact mGBA save state.
    """
    data = path if isinstance(path, bytes) else path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("checkpoint is not a PNG/mGBA save state")
    state = save_ram = None
    offset = 8
    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError("truncated mGBA save state")
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + payload) != crc:
            raise ValueError(f"bad {kind!r} checksum")
        try:
            if kind == b"gbAs":
                state = zlib.decompress(payload)
            elif kind == b"gbAx" and payload[:4] == b"\x02\0\0\0":
                save_ram = zlib.decompress(payload[8:])
        except zlib.error as exc:
            raise ValueError("invalid compressed mGBA save state") from exc
        offset = end
    if state is None or save_ram is None:
        raise ValueError("mGBA state or save RAM is missing")
    return Checkpoint(state, save_ram)


def read_checkpoint(path: Path) -> Checkpoint:
    """Read an mGBA .ss0 save state, or a raw core state from any other file."""
    return read_ss0(path) if path.suffix.lower() == ".ss0" else Checkpoint(path.read_bytes())


def restore(game: GbaEmulator, checkpoint: Checkpoint) -> None:
    """Load a checkpoint and advance one tick so the core presents it."""
    if checkpoint.save_ram is not None:
        game.load_save_ram(checkpoint.save_ram)
    game.load_state(checkpoint.state)
    game.step()


def perform(game: GbaEmulator, action: GameAction, hold_ticks: int,
            release_ticks: int) -> None:
    """Hold the action's input for hold_ticks, then release it for release_ticks."""
    match action:
        case Press(button):
            game.step(hold_ticks, (button,))
        case Wait():
            game.step(hold_ticks)
        case _:
            assert_never(action)
    game.step(release_ticks)


def _visible_cursor_frame(first: Frame, second: Frame) -> Frame:
    """Return whichever frame is brighter at the strongest local blink."""
    if (first.width, first.height) != (second.width, second.height):
        return second
    blocks: dict[tuple[int, int], int] = {}
    for index in range(0, len(first.rgba), 4):
        before = first.rgba[index:index + 3]
        after = second.rgba[index:index + 3]
        if sum(abs(a - b) for a, b in zip(before, after, strict=True)) <= BLINK_PIXEL_DELTA:
            continue
        pixel = index // 4
        block = (pixel % first.width // BLINK_BLOCK_PIXELS,
                 pixel // first.width // BLINK_BLOCK_PIXELS)
        blocks[block] = blocks.get(block, 0) + sum(after) - sum(before)
    # ponytail: local brightness is a visual heuristic; use cursor detection if another
    # menu animation dominates it.
    strongest = max(blocks.values(), key=abs, default=0)
    return second if strongest >= 0 else first


class GameplaySession:
    """Run the emulator for the agent and save every captured frame as a screenshot."""

    def __init__(self, emulator: GbaEmulator, output: Path,
                 hold_ticks: int = HOLD_TICKS, release_ticks: int = RELEASE_TICKS):
        if min(hold_ticks, release_ticks) < 1:
            raise ValueError("tick counts must be positive")
        self.emulator = emulator
        self.output = output
        self.hold_ticks = hold_ticks
        self.release_ticks = release_ticks
        self.image_index = -1
        self.action_count = 0
        self.tick = 0
        self.current: Frame | None = None
        self.current_name = ""
        self.current_game_state: GameState | None = None
        self.warning: str | None = None
        self._last_action: GameAction | None = None
        self._unchanged_count = 0
        # Test doubles aren't GbaEmulators and run without RAM-timed captures.
        self.ram = GameRam(emulator) if isinstance(emulator, GbaEmulator) else None

    def ram_state(self) -> RamState | None:
        return self.ram.state() if self.ram is not None else None

    def game_state(self) -> GameState | None:
        """Return the game state the model may know, or None without RAM."""
        if self.ram is None:
            return None
        position = self.ram.player_position()
        characters = () if position is None else tuple(
            Character(self.ram.species_name(npc.species), npc.position, npc.direction)
            for npc in self.ram.ground_characters()
            if abs(npc.position[0] - position[0]) <= VISIBLE_X
            and abs(npc.position[1] - position[1]) <= VISIBLE_Y)
        return GameState(position, self.ram.player_tile(), self.ram.player_direction(),
                         characters)

    def load(self, checkpoint: Checkpoint) -> None:
        """Restore a checkpoint and start counting ticks and actions again.

        Raises:
            Exception: the core's error for an unusable checkpoint, after the
                previous game state has been restored.
        """
        previous_state = self.emulator.save_state()
        previous_ram = self.emulator.save_ram() if checkpoint.save_ram is not None else None
        try:
            restore(self.emulator, checkpoint)
        except Exception:
            if previous_ram is not None:
                self.emulator.load_save_ram(previous_ram)
            self.emulator.load_state(previous_state)
            raise
        self.tick = 1
        self.action_count = 0
        self.warning = None
        self._last_action = None
        self._unchanged_count = 0
        self.capture()

    def wait_for_text(self) -> None:
        """Run without input until the current text page is fully printed, then capture it."""
        if self.current is None:
            raise RuntimeError("capture a frame before waiting")
        state = self.ram_state()
        waited = 0
        while state is not None and state.text_printing and waited < TEXT_WAIT_LIMIT:
            ticks = min(TEXT_POLL_TICKS, TEXT_WAIT_LIMIT - waited)
            self.emulator.step(ticks)
            self.tick += ticks
            waited += ticks
            state = self.ram_state()
        if waited:
            if state is not None and not state.text_printing and state.mode == "menu":
                self.emulator.step(MENU_AFTER_TEXT_TICKS)
                self.tick += MENU_AFTER_TEXT_TICKS
            self.capture()

    def capture(self, frame: Frame | None = None) -> Frame:
        """Make the current or supplied frame the session's current frame and save it."""
        self.image_index += 1
        self.current = frame if frame is not None else self.emulator.screenshot()
        self.current_game_state = self.game_state()
        self.current_name = f"frames/{self.image_index:06d}.png"
        (self.output / "frames").mkdir(parents=True, exist_ok=True)
        self.current.save_png(self.output / self.current_name)
        return self.current

    def idle(self, ticks: int = 1) -> None:
        """Run ticks without input; the current frame stays the last one captured."""
        self.emulator.step(ticks)
        self.tick += ticks

    def current_frame(self) -> Frame:
        """Return the current frame; in menus, wait through a cursor blink to show the cursor."""
        if self.current is None:
            raise RuntimeError("capture a frame before observing")
        state = self.ram_state()
        if state is None or state.mode != "menu":
            return self.current
        first = self.current
        self.emulator.step(CURSOR_BLINK_TICKS)
        self.tick += CURSOR_BLINK_TICKS
        second = self.emulator.screenshot()
        if (next_state := self.ram_state()) is None or next_state.mode != "menu":
            return self.capture(second)
        return self.capture(_visible_cursor_frame(first, second))

    def observe(self) -> Observation:
        """Return a readable frame without skipping cutscenes or advancing dialogue."""
        self.wait_for_text()
        frame = self.current_frame()
        return Observation(frame, self.current_name, self.current_game_state)

    def apply(self, action: GameAction) -> Transition:
        """Perform ``action``, capture the result and update the stall warning."""
        if self.current is None:
            raise RuntimeError("capture a frame before applying an action")
        before = Observation(self.current, self.current_name, self.current_game_state)
        tick_before = self.tick
        perform(self.emulator, action, self.hold_ticks, self.release_ticks)
        self.tick += self.hold_ticks + self.release_ticks
        after = self.capture()
        self.action_count += 1
        if isinstance(action, Press) and scene_distance(before.frame, after) < STALL_SCENE_DISTANCE:
            self._unchanged_count = self._unchanged_count + 1 if action == self._last_action else 1
        else:
            self._unchanged_count = 0
        self._last_action = action
        stalled = self._unchanged_count >= STALL_PRESSES
        self.warning = (f"Possible stall: little screen change after {self._unchanged_count} "
                        f"{name(action)} presses" if stalled else None)
        return Transition(before, Observation(after, self.current_name, self.current_game_state),
                          self.image_index - 1,
                          self.action_count, tick_before, self.tick)

    def save_state(self, filename: str) -> None:
        """Write the core state to states/<filename>, replacing any earlier file atomically."""
        states = self.output / "states"
        states.mkdir(parents=True, exist_ok=True)
        temporary = states / f"{filename}.tmp"
        temporary.write_bytes(self.emulator.save_state())
        temporary.replace(states / filename)
