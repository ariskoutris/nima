"""Read game state from RAM for the matching Red Rescue Team GBA ROM."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from emulator import GbaEmulator

ROM_SHA1 = "9f4cfc5b5f4859d17169a485462e977c7aac2b89"

# Addresses from https://github.com/pret/pmd-red for ROM_SHA1.
MAIN_LOOP = 0x0203B03C  # sMainLoopsUnk: 1 dungeon, 2 ground
TEXTBOX = 0x03001B64  # sTextbox
TEXT_PHASE = 0x0202E778  # observed text printer phase: 1 while characters stream, 2 at page end
DIALOGUE_TEXT = 0x0202E798  # sDialogueTextBuffer, the formatted text of the open textbox
DIALOGUE_TEXT_SIZE = 1000
GROUND_LIVES = 0x03001B84  # gGroundLives
GROUND_LIVE_COUNT, GROUND_LIVE_SIZE = 24, 0x1F0
GROUND_MAP_ACTION = 0x03001B6C  # gGroundMapAction -> groundMapId at +0xE4
GROUND_LIVE_POSITION, GROUND_LIVE_DIRECTION = 0x144, 0x142  # GroundLive unk144, direction1
GROUND_LIVE_SPECIES = 0x8
# The species table in ROM: one entry per species id, each starting with a pointer to the
# species name. Found from the pointers to "Wigglytuff", "Granbull" and "Lombre" (ids 40, 235
# and 296), which sit exactly SPECIES_ENTRY_SIZE apart per id.
SPECIES_TABLE, SPECIES_ENTRY_SIZE = 0x08357B98, 0x48
SPECIES_NAME_SIZE = 16  # longest name read; names end at the first zero byte
# pmd-red names gDungeon without an address; this is the EWRAM_INIT pointer whose team slot 0
# points at its own teamPokemonEntities, found by scanning RAM in normal-maze.ss0.
DUNGEON = 0x0203B418  # gDungeon
TEAM_POKEMON, TEAM_SIZE = 0x1357C, 4  # Dungeon.teamPokemon
ENTITY_MONSTER = 1
ENTITY_POSITION, ENTITY_INFO = 0x4, 0x70  # Entity.pos (tiles), Entity.axObj.info.monster
INFO_IS_LEADER, INFO_DIRECTION = 0x7, 0x46  # EntityInfo.isTeamLeader, .action.direction
# enum Direction. The cardinal values were measured on square.ss0 and normal-maze.ss0 by
# walking each way; the diagonals follow the enum order. The game masks direction bytes
# with DIRECTION_MASK because higher bits can carry flags.
DIRECTION_MASK = 7
DIRECTIONS = ("south", "southeast", "east", "northeast",
              "north", "northwest", "west", "southwest")

# Formatting codes in dialogue text: "~27" is a hex-escaped character, "#c\x06" and
# "#C5" set a colour, and other "#X" codes reset colour or pause printing.
_ESCAPED = re.compile(r"~([0-9a-fA-F]{2})")
_CODE = re.compile(r"#[cC].|#[A-Za-z]", re.DOTALL)


@cache
def _rom_sha1(rom: Path) -> str:
    return hashlib.sha1(rom.read_bytes()).hexdigest()  # one hash per ROM path, not per session


def _ewram(pointer: int, size: int) -> bool:
    return 0x02000000 <= pointer and pointer + size <= 0x02040000


def _position(raw: bytes) -> tuple[int, int]:
    x, y = struct.unpack("<ii", raw)  # PixelPos, s24.8
    return x // 256, y // 256


@dataclass(frozen=True)
class GroundCharacter:
    """A character loaded on the ground map, positioned in pixels like the leader."""

    species: int
    position: tuple[int, int]
    direction: str


@dataclass(frozen=True)
class RamState:
    mode: str
    textbox_type: int | None
    text_printing: bool
    speaker_species: int | None
    player_position: tuple[int, int] | None
    player_tile: tuple[int, int] | None
    player_direction: str | None


class GameRam:
    """Read game state from RAM.

    Raises:
        ValueError: when the emulator's ROM isn't the one these addresses match.
    """

    def __init__(self, game: GbaEmulator) -> None:
        if _rom_sha1(game.rom) != ROM_SHA1:
            raise ValueError(f"RAM addresses require the Red Rescue Team ROM {ROM_SHA1}")
        self._game = game

    def _value(self, address: int, fmt: str) -> int:
        return struct.unpack(fmt, self._game.read_memory(address, struct.calcsize(fmt)))[0]

    def _textbox(self) -> int | None:
        textbox = self._value(TEXTBOX, "<I")
        return textbox if _ewram(textbox, 0x464) else None

    def _ground_lives(self) -> int | None:
        if self._value(MAIN_LOOP, "<I") != 2:
            return None
        lives = self._value(GROUND_LIVES, "<I")
        return lives if _ewram(lives, GROUND_LIVE_COUNT * GROUND_LIVE_SIZE) else None

    def _dungeon_leader(self) -> tuple[int, int] | None:
        """Return the leader's Entity and EntityInfo addresses, or None outside dungeons."""
        if self._value(MAIN_LOOP, "<I") != 1:
            return None
        dungeon = self._value(DUNGEON, "<I")
        if not _ewram(dungeon, TEAM_POKEMON + TEAM_SIZE * 4):
            return None
        for slot in range(TEAM_SIZE):
            entity = self._value(dungeon + TEAM_POKEMON + slot * 4, "<I")
            if not _ewram(entity, ENTITY_INFO + 4) or self._value(entity, "<I") != ENTITY_MONSTER:
                continue
            info = self._value(entity + ENTITY_INFO, "<I")
            if _ewram(info, INFO_DIRECTION + 1) and self._value(info + INFO_IS_LEADER, "<B"):
                return entity, info
        return None

    def textbox_type(self) -> int | None:
        """Return the open textbox's type (1 dialogue, 4 dialogue with a menu), or None."""
        textbox = self._textbox()
        return self._value(textbox, "<I") if textbox is not None else None

    def text_printing(self) -> bool:
        return self.textbox_type() in (1, 4) and self._value(TEXT_PHASE, "<I") == 1

    def textbox_text(self) -> str:
        """Return the open textbox's whole page, complete even while it prints; "" when closed."""
        if self.textbox_type() in (None, 0):
            return ""
        raw = self._game.read_memory(DIALOGUE_TEXT, DIALOGUE_TEXT_SIZE).split(b"\0")[0]
        text = _ESCAPED.sub(lambda match: chr(int(match[1], 16)), raw.decode("latin-1"))
        return _CODE.sub("", text)

    def speaker(self) -> int | None:
        """Return the species in the open textbox's speaker portrait, or None."""
        textbox = self._textbox()
        return self._value(textbox + 0x462, "<h") if textbox is not None else None

    def player_position(self) -> tuple[int, int] | None:
        """Return the leader's ground position in pixels, or None outside ground maps."""
        lives = self._ground_lives()
        if lives is None:
            return None
        return _position(self._game.read_memory(lives + GROUND_LIVE_POSITION, 8))

    def player_species(self) -> int | None:
        """Return the leader's species on ground maps, or None elsewhere or before it spawns."""
        lives = self._ground_lives()
        if lives is None or self._value(lives + 2, "<h") == -1:
            return None
        return self._value(lives + GROUND_LIVE_SPECIES, "<h")

    def player_tile(self) -> tuple[int, int] | None:
        """Return the leader's dungeon tile as (x, y), or None outside dungeons."""
        leader = self._dungeon_leader()
        if leader is None:
            return None
        x, y = struct.unpack("<hh", self._game.read_memory(leader[0] + ENTITY_POSITION, 4))
        return x, y

    def player_direction(self) -> str | None:
        """Return the way the leader faces on ground maps and in dungeons, or None elsewhere."""
        if (lives := self._ground_lives()) is not None:
            if self._value(lives + 2, "<h") == -1:  # the leader's live isn't spawned yet
                return None
            direction = self._value(lives + GROUND_LIVE_DIRECTION, "<B")
        elif (leader := self._dungeon_leader()) is not None:
            direction = self._value(leader[1] + INFO_DIRECTION, "<B")
        else:
            return None
        return DIRECTIONS[direction & DIRECTION_MASK]

    def npc_position(self, species: int) -> tuple[int, int] | None:
        """Return the ground position in pixels of the first loaded ``species``, or None."""
        return next((character.position for character in self.ground_characters()
                     if character.species == species), None)

    def ground_characters(self) -> tuple[GroundCharacter, ...]:
        """Return the characters loaded on the ground map besides the leader; () elsewhere."""
        lives = self._ground_lives()
        characters = []
        for index in range(1, GROUND_LIVE_COUNT if lives is not None else 0):
            entry = lives + index * GROUND_LIVE_SIZE
            if self._value(entry + 2, "<h") == -1:  # an empty slot
                continue
            direction = self._value(entry + GROUND_LIVE_DIRECTION, "<B")
            characters.append(GroundCharacter(
                self._value(entry + GROUND_LIVE_SPECIES, "<h"),
                _position(self._game.read_memory(entry + GROUND_LIVE_POSITION, 8)),
                DIRECTIONS[direction & DIRECTION_MASK]))
        return tuple(characters)

    def species_name(self, species: int) -> str:
        """Return a species' name from the ROM's species table."""
        pointer = self._value(SPECIES_TABLE + species * SPECIES_ENTRY_SIZE, "<I")
        return self._game.read_memory(pointer, SPECIES_NAME_SIZE).split(b"\0")[0].decode("latin-1")

    def ground_map(self) -> int | None:
        """Return the current ground map id (1 Pokémon Square, 3 Whiscash Pond), or None."""
        action = self._value(GROUND_MAP_ACTION, "<I")
        return self._value(action + 0xE4, "<h") if _ewram(action, 0xE6) else None

    def state(self) -> RamState:
        """Return a snapshot for timing captures, the agent's game state and the operator."""
        main_loop = self._value(MAIN_LOOP, "<I")
        textbox_type = self.textbox_type()
        printing = self.text_printing()
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
        speaker = self.speaker() if mode == "dialogue" else None
        return RamState(mode, textbox_type, printing, speaker, self.player_position(),
                        self.player_tile(), self.player_direction())
