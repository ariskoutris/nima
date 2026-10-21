"""Small Game Boy Advance adapter for a Libretro core, stepped one tick at a time."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import struct
import zlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

from libretro import Session
from libretro.api.input import JoypadState
from libretro.api.memory import RETRO_MEMORY_SAVE_RAM
from libretro.drivers.video.software.array import ArrayVideoDriver

BUTTONS = frozenset("A B SELECT START UP DOWN LEFT RIGHT L R".split())


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    rgba: bytes

    @cached_property
    def data_url(self) -> str:
        """The PNG as a base64 data URL, encoded once per frame for model requests."""
        return "data:image/png;base64," + base64.b64encode(self.png_bytes()).decode("ascii")

    def png_bytes(self) -> bytes:
        """Encode the frame as a PNG without an image dependency."""
        def chunk(kind: bytes, data: bytes) -> bytes:
            return (struct.pack(">I", len(data)) + kind + data
                    + struct.pack(">I", zlib.crc32(kind + data)))

        stride = self.width * 4
        rows = b"".join(b"\0" + self.rgba[y:y + stride]
                        for y in range(0, len(self.rgba), stride))
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 6, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(rows))
               + chunk(b"IEND", b""))
        return png

    def save_png(self, path: str | Path) -> None:
        Path(path).write_bytes(self.png_bytes())


class GbaEmulator:
    """Run a GBA ROM in mGBA, one controlled batch of ticks at a time."""

    def __init__(self, rom: str | Path, core: str | Path):
        self.rom = Path(rom).resolve()
        self.core_path = Path(core).resolve()
        if not self.rom.is_file():
            raise FileNotFoundError(f"ROM not found: {self.rom}")
        if not self.core_path.is_file():
            raise FileNotFoundError(f"Libretro core not found: {self.core_path}")
        self._buttons: set[str] = set()
        self.on_frame: Callable[[Frame], None] | None = None
        self._video = ArrayVideoDriver()
        # No core log: libretro.py can't expand printf arguments, so it would print bare "%s: %s".
        self._session = Session(self.core_path, self.rom, input=self._input(), video=self._video,
                                log=None)

    def _input(self):
        while True:
            yield JoypadState(**{button.lower(): True for button in self._buttons})

    def __enter__(self) -> GbaEmulator:
        self._session.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._session.__exit__(*args)

    def step(self, ticks: int = 1, buttons: tuple[str, ...] = ()) -> None:
        """Run ``ticks`` ticks holding ``buttons``, then release them.

        Raises:
            ValueError: when ``ticks`` is not positive or a button is not a GBA button.
        """
        if ticks < 1:
            raise ValueError("ticks must be positive")
        normalized = {button.upper() for button in buttons}
        if invalid := normalized - BUTTONS:
            raise ValueError(f"invalid GBA button(s): {', '.join(sorted(invalid))}")
        self._buttons = normalized
        try:
            for _ in range(ticks):
                self._session.run()
                if self.on_frame:
                    self.on_frame(self.screenshot())
        finally:
            self._buttons.clear()

    def screenshot(self) -> Frame:
        shot = self._video.screenshot()
        if shot is None:
            raise RuntimeError("no frame yet; call step first")
        return Frame(shot.width, shot.height, bytes(shot.data))

    def save_state(self) -> bytes:
        state = bytearray(self._session.core.serialize_size())
        if not state or not self._session.core.serialize(state):
            raise RuntimeError("core could not serialize its state")
        return bytes(state)

    def load_state(self, state: bytes) -> None:
        self._buttons.clear()
        if not self._session.core.unserialize(state):
            raise RuntimeError("core could not restore this state")

    def save_ram(self) -> bytes:
        """Return the game's battery save, suitable for a .sav file."""
        memory = self._session.core.get_memory(RETRO_MEMORY_SAVE_RAM)
        if memory is None:
            raise RuntimeError("core has no save RAM")
        return bytes(memory)

    def read_memory(self, address: int, size: int) -> bytes:
        """Read a GBA address through the core's Libretro memory map."""
        maps = self._session.memory_maps
        if maps is None or size < 1:
            raise ValueError("memory map unavailable or invalid read size")
        for index in range(maps.num_descriptors):
            region = maps.descriptors[index]
            if (region.ptr and region.start <= address and
                    address + size <= region.start + region.len):
                return ctypes.string_at(region.ptr.value + region.offset + address - region.start,
                                        size)
        raise ValueError(f"unmapped GBA memory: {address:#x} ({size} bytes)")

    def load_save_ram(self, data: bytes) -> None:
        """Load a battery save after entering the emulator session."""
        memory = self._session.core.get_memory(RETRO_MEMORY_SAVE_RAM)
        if memory is None:
            raise RuntimeError("core has no save RAM")
        if len(data) != len(memory):
            raise ValueError(f"save RAM is {len(memory)} bytes, got {len(data)}")
        memory[:] = data


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a GBA ROM with the mGBA adapter")
    parser.add_argument("rom", type=Path)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--boot-ticks", type=int, default=120)
    parser.add_argument("--screenshot", type=Path, default=Path("runs/smoke.png"))
    args = parser.parse_args()

    with GbaEmulator(args.rom, args.core) as emulator:
        emulator.step(args.boot_ticks)
        checkpoint = emulator.save_state()
        emulator.step(15)
        no_input = emulator.screenshot()
        emulator.load_state(checkpoint)
        emulator.step(15, ("A",))
        after = emulator.screenshot()
        emulator.load_state(checkpoint)
        emulator.step(15, ("A",))
        replay = emulator.screenshot()
        assert after.rgba == replay.rgba, "save-state replay diverged"
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        after.save_png(args.screenshot)
        print(f"frame: {after.width}x{after.height}")
        print(f"A changed image versus no input: {no_input.rgba != after.rgba}")
        print(f"replay SHA-256: {hashlib.sha256(replay.rgba).hexdigest()}")
        print(f"screenshot: {args.screenshot}")


if __name__ == "__main__":
    main()
