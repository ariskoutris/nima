"""Small frame-by-frame Game Boy Advance adapter for a Libretro core."""

from __future__ import annotations

import argparse
import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from libretro import Session
from libretro.api.input import JoypadState
from libretro.api.memory import RETRO_MEMORY_SAVE_RAM
from libretro.drivers.video.software.array import ArrayVideoDriver


BUTTONS = frozenset("A B SELECT START UP DOWN LEFT RIGHT L R".split())
DEFAULT_CORE = Path(__file__).parent / ".local/cores/mgba_libretro.dylib"


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    rgba: bytes

    def png_bytes(self) -> bytes:
        """Encode an RGBA screenshot without an image dependency."""
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
    """Run a GBA ROM in mGBA, one controlled batch of frames at a time."""

    def __init__(self, rom: str | Path, core: str | Path = DEFAULT_CORE):
        self.rom = Path(rom).resolve()
        self.core_path = Path(core).resolve()
        if not self.rom.is_file():
            raise FileNotFoundError(f"ROM not found: {self.rom}")
        if not self.core_path.is_file():
            raise FileNotFoundError(f"Libretro core not found: {self.core_path}")
        self._buttons: set[str] = set()
        self._video = ArrayVideoDriver()
        self._session = Session(self.core_path, self.rom, input=self._input(), video=self._video)

    def _input(self):
        while True:
            yield JoypadState(**{button.lower(): True for button in self._buttons})

    def __enter__(self) -> GbaEmulator:
        self._session.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._session.__exit__(*args)

    def step(self, frames: int = 1, buttons: tuple[str, ...] = ()) -> None:
        if frames < 1:
            raise ValueError("frames must be positive")
        normalized = {button.upper() for button in buttons}
        if invalid := normalized - BUTTONS:
            raise ValueError(f"invalid GBA button(s): {', '.join(sorted(invalid))}")
        self._buttons = normalized
        try:
            for _ in range(frames):
                self._session.run()
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
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--boot-frames", type=int, default=120)
    parser.add_argument("--screenshot", type=Path, default=Path(".local/smoke.png"))
    args = parser.parse_args()

    with GbaEmulator(args.rom, args.core) as emulator:
        emulator.step(args.boot_frames)
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
