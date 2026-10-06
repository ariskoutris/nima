"""Observed RAM state for the matching Red Rescue Team GBA ROM."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from emulator import GbaEmulator

ROM_SHA1 = "9f4cfc5b5f4859d17169a485462e977c7aac2b89"


def _ewram(pointer: int, size: int) -> bool:
    return 0x02000000 <= pointer and pointer + size <= 0x02040000


def _u32(game: GbaEmulator, address: int) -> int:
    return struct.unpack("<I", game.read_memory(address, 4))[0]


@dataclass(frozen=True)
class RamState:
    mode: str
    textbox_type: int | None
    text_printing: bool
    speaker_species: int | None
    player_position: tuple[int, int] | None


def read_ram_state(game: GbaEmulator) -> RamState:
    """Read a snapshot without advancing the emulator.

    Addresses match ROM_SHA1. Callers should check the ROM before using them.
    """
    main_loop = _u32(game, 0x0203B03C)  # sMainLoopsUnk: 1 dungeon, 2 ground
    textbox = _u32(game, 0x03001B64)  # sTextbox
    textbox_type = _u32(game, textbox) if _ewram(textbox, 0x464) else None
    # Observed text printer phase: 1 while characters stream, 2 at the page end.
    printing = textbox_type in (1, 4) and _u32(game, 0x0202E778) == 1

    lives = _u32(game, 0x03001B84)  # gGroundLives
    position = None
    if main_loop == 2 and _ewram(lives, 0x14C):
        x, y = struct.unpack("<ii", game.read_memory(lives + 0x144, 8))
        position = x // 256, y // 256

    if main_loop == 1:
        mode = "dungeon"
    elif main_loop != 2:
        mode = "unknown"
    elif textbox_type == 1 or printing:
        mode = "dialogue"
    elif textbox_type == 4:
        # Type 4 also carries shop dialogue; the phase only distinguishes active printing.
        mode = "menu"
    else:
        mode = "field"
    speaker = (struct.unpack("<h", game.read_memory(textbox + 0x462, 2))[0]
               if mode == "dialogue" else None)
    return RamState(mode, textbox_type, printing, speaker, position)
