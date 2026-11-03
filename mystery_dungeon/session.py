"""Emulator I/O: load checkpoints, observe frames, and apply actions."""

from __future__ import annotations

import struct
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, assert_never

from emulator import Frame, GbaEmulator
from mystery_dungeon.actions import GameAction, Press, Wait
from mystery_dungeon.ram_state import GameRam, RamState

TICK_SECONDS = 1 / 60  # one tick at the GBA's refresh rate
HOLD_TICKS = 8  # ticks each action holds its input
RELEASE_TICKS = 4  # ticks with no input after each action
TEXT_WAIT_LIMIT = 300  # most ticks spent waiting for one page of text to print
# Menus that follow dialogue (e.g. the shop's choices) open a few ticks after the last glyph.
MENU_AFTER_TEXT_TICKS = 12
# The menu cursor blinks 8 ticks on and 8 off, measured in the shop and dungeon menus; waiting
# one full blink shows it from any phase.
CURSOR_BLINK_TICKS = 16
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
    species: str | None = None  # the player's, on ground maps


@dataclass(frozen=True)
class Observation:
    """A frame, the path of its screenshot relative to the session output, and the game state
    when it was captured."""

    frame: Frame
    screenshot: str
    game_state: GameState


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
        self._last_observation: Observation | None = None
        self._observed_tick = 0
        self.ram = GameRam(emulator)

    @property
    def last_observation(self) -> Observation:
        """The latest observation recorded by load, observe or apply.

        Raises:
            RuntimeError: before the first one.
        """
        if self._last_observation is None:
            raise RuntimeError("no observation yet: load or observe first")
        return self._last_observation

    def ram_state(self) -> RamState:
        return self.ram.state()

    def game_state(self) -> GameState:
        """Return the game state the model may know."""
        position = self.ram.player_position()
        characters = () if position is None else tuple(
            Character(self.ram.species_name(npc.species), npc.position, npc.direction)
            for npc in self.ram.ground_characters()
            if abs(npc.position[0] - position[0]) <= VISIBLE_X
            and abs(npc.position[1] - position[1]) <= VISIBLE_Y)
        species = self.ram.player_species()
        return GameState(position, self.ram.player_tile(), self.ram.player_direction(),
                         characters, None if species is None else self.ram.species_name(species))

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
        self._record_observation()

    def _record_observation(self) -> Observation:
        """Record the screen and game state as the last observation, and save the screenshot."""
        self.image_index += 1
        frame = self.emulator.screenshot()
        screenshot = f"frames/{self.image_index:06d}.png"
        (self.output / "frames").mkdir(parents=True, exist_ok=True)
        frame.save_png(self.output / screenshot)
        self._last_observation = Observation(frame, screenshot, self.game_state())
        self._observed_tick = self.tick
        return self._last_observation

    def idle(self, ticks: int = 1) -> None:
        """Run ticks without input; the next observe records what they did."""
        self.emulator.step(ticks)
        self.tick += ticks

    def observe(self) -> Observation:
        """
        Return the current frame and game state.

        Waits for text to print fully and the blinking cursor to be visible.
        """
        self._idle_until(lambda: not self.ram.state().text_printing, TEXT_WAIT_LIMIT)
        if self.ram.state().mode == "menu":
            self._idle_until(self.ram.menu_cursor_shown, MENU_AFTER_TEXT_TICKS + CURSOR_BLINK_TICKS)
        if self._last_observation is None or self._observed_tick != self.tick:
            return self._record_observation()
        return self._last_observation

    def _idle_until(self, done: Callable[[], bool], limit: int) -> None:
        """Idle until ``done()`` holds or ``limit`` ticks pass."""
        for _ in range(limit):
            if done():
                return
            self.idle()

    def apply(self, action: GameAction) -> Transition:
        """Perform ``action`` and observe the result without waiting."""
        before = self.last_observation
        tick_before = self.tick
        perform(self.emulator, action, self.hold_ticks, self.release_ticks)
        self.tick += self.hold_ticks + self.release_ticks
        after = self._record_observation()
        self.action_count += 1
        return Transition(before, after, self.image_index - 1,
                          self.action_count, tick_before, self.tick)

    def save_state(self, filename: str) -> None:
        """Write the core state to states/<filename>, replacing any earlier file atomically."""
        states = self.output / "states"
        states.mkdir(parents=True, exist_ok=True)
        temporary = states / f"{filename}.tmp"
        temporary.write_bytes(self.emulator.save_state())
        temporary.replace(states / filename)


class RealtimeSession:
    """Keep the game running without input, at real speed, from each observation until the
    next action.

    The game doesn't wait for the model, so an action meets the game as it is when the
    reply arrives, not as the model saw it.
    """

    def __init__(self, session: GameplaySession) -> None:
        self._session = session
        self._halted = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def observe(self) -> Observation:
        self.halt()  # one idle thread at a time
        observation = self._session.observe()
        self._halted.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return observation

    def apply(self, action: GameAction) -> Transition:
        self.halt()
        return self._session.apply(action)

    def halt(self) -> None:
        """Stop the game after its current tick.

        Raises:
            Exception: the emulator's error if a tick failed while the game ran.
        """
        self._halted.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if (error := self._error) is not None:
            self._error = None
            raise error

    def _run(self) -> None:
        started, ticks = time.monotonic(), 0
        try:
            # Each tick waits for its time, so the game runs as many ticks as the GBA would.
            while not self._halted.wait(max(0.0, started + ticks * TICK_SECONDS
                                            - time.monotonic())):
                self._session.idle()
                ticks += 1
        except Exception as exc:
            self._error = exc
