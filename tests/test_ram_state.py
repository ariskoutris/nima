"""RAM observations against the checked-in Red Rescue Team checkpoints."""

import struct

import pytest
from fakes import ScriptedModel, press

from emulator import GbaEmulator
from mystery_dungeon.actions import Press, Stop
from mystery_dungeon.agent import Agent
from mystery_dungeon.benchmark import CHECKPOINTS, CORE, ROM
from mystery_dungeon.ram_state import (
    DUNGEON,
    GROUND_LIVE_DIRECTION,
    GROUND_LIVES,
    TEAM_POKEMON,
    GameRam,
)
from mystery_dungeon.session import GameplaySession, read_ss0, restore


@pytest.mark.parametrize("checkpoint,ready_mode", [
    ("face-npc", "dialogue"), ("shop-interact", "menu"),
])
def test_dialogue_waits_for_full_page(tmp_path, checkpoint, ready_mode):
    if not all(path.is_file() for path in
               (ROM, CORE, CHECKPOINTS / f"{checkpoint}.ss0")):
        pytest.skip("local ROM, core, and checkpoint required")
    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / f"{checkpoint}.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()
        assert GameRam(game).state().mode == "field"
        session.apply(Press("A"))
        initial = GameRam(game).state()
        assert initial.mode == "dialogue" and initial.text_printing
        partial = session.current.rgba
        actions = session.action_count
        ticks = session.tick
        session.wait_for_text()
        assert session.tick > ticks
        assert session.action_count == actions
        assert GameRam(game).state().mode == ready_mode
        assert not GameRam(game).state().text_printing
        if ready_mode == "menu":  # the snapshot names a speaker only in dialogue
            assert GameRam(game).state().speaker_species is None
        assert session.current.rgba != partial


def test_field_menu_and_dungeon_modes():
    paths = (ROM, CORE, CHECKPOINTS / "square.ss0", CHECKPOINTS / "normal-maze.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoints required")
    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "square.ss0"))
        field = GameRam(game).state()
        assert field.mode == "field" and field.player_position is not None
        game.step(8, ("B",))
        game.step(4)
        assert GameRam(game).state().mode == "menu"
        restore(game, read_ss0(CHECKPOINTS / "normal-maze.ss0"))
        assert GameRam(game).state().mode == "dungeon"


@pytest.mark.parametrize("checkpoint,read", [
    ("square", GameRam.player_position), ("normal-maze", GameRam.player_tile),
])
def test_player_direction_follows_the_dpad(checkpoint, read):
    paths = (ROM, CORE, CHECKPOINTS / f"{checkpoint}.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoint required")
    with GbaEmulator(ROM, CORE) as game:
        ram = GameRam(game)
        restore(game, read_ss0(CHECKPOINTS / f"{checkpoint}.ss0"))
        assert ram.player_direction() == "south"
        for button, direction, axis, sign in (
            ("RIGHT", "east", 0, 1), ("UP", "north", 1, -1),
            ("LEFT", "west", 0, -1), ("DOWN", "south", 1, 1),
        ):
            before = read(ram)
            game.step(20, (button,))
            game.step(40)
            after = read(ram)
            assert ram.player_direction() == direction
            assert (after[axis] - before[axis]) * sign > 0
        state = ram.state()
        assert state.player_direction == "south"
        assert (state.player_position is None) == (checkpoint == "normal-maze")
        assert (state.player_tile is None) == (checkpoint == "square")


class PatchedRam:
    """A real emulator whose memory reads see ``patches`` laid over RAM."""

    def __init__(self, game: GbaEmulator, patches: dict[int, bytes]) -> None:
        self.rom = game.rom
        self._game = game
        self._patches = patches

    def read_memory(self, address: int, size: int) -> bytes:
        data = bytearray(self._game.read_memory(address, size))
        for start, raw in self._patches.items():
            for offset, byte in enumerate(raw):
                if 0 <= start + offset - address < size:
                    data[start + offset - address] = byte
        return bytes(data)


def _u32(game: GbaEmulator, address: int) -> int:
    return struct.unpack("<I", game.read_memory(address, 4))[0]


def test_player_direction_masks_flag_bits_and_skips_unspawned_leader():
    paths = (ROM, CORE, CHECKPOINTS / "square.ss0", CHECKPOINTS / "normal-maze.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoints required")
    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "square.ss0"))
        lives = _u32(game, GROUND_LIVES)
        flagged = PatchedRam(game, {lives + GROUND_LIVE_DIRECTION: b"\xf2"})
        assert GameRam(flagged).player_direction() == "east"
        unspawned = PatchedRam(game, {lives + 2: struct.pack("<h", -1)})
        assert GameRam(unspawned).player_direction() is None

        restore(game, read_ss0(CHECKPOINTS / "normal-maze.ss0"))
        stale = PatchedRam(game, {DUNGEON: struct.pack("<I", 0x02000000 - TEAM_POKEMON)})
        assert GameRam(stale).player_direction() is None
        assert GameRam(stale).player_tile() is None


def test_menu_observation_shows_blinking_cursor_without_an_action(tmp_path):
    paths = (ROM, CORE, CHECKPOINTS / "shop-interact.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoint required")

    def light_pixels(frame, rect):
        left, top, right, bottom = rect
        return sum(min(frame.rgba[index:index + 3]) > 180
                   for y in range(top, bottom) for x in range(left, right)
                   for index in ((y * frame.width + x) * 4,))

    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "shop-interact.ss0"))
        press(game, "A")
        while GameRam(game).state().text_printing:
            game.step(4)
        game.step(12)
        popup = game.save_state()

        press(game, "DOWN")
        press(game, "A")
        while GameRam(game).state().text_printing:
            game.step(4)
        game.step(12)
        press(game, "A")
        game.step(12)
        listing = game.save_state()
        for _ in range(4):
            press(game, "DOWN")
        pidgey = game.save_state()

        for name, state, rect, threshold in (
            ("popup", popup, (182, 65, 193, 78), 8),
            ("list", listing, (14, 30, 26, 44), 0),
            ("pidgey", pidgey, (14, 78, 26, 91), 0),
        ):
            for phase in range(16):
                game.load_state(state)
                game.step(phase + 1)
                session = GameplaySession(game, tmp_path / name / str(phase))
                session.capture()
                assert GameRam(game).state().mode == "menu"
                observed = session.current_frame()
                assert light_pixels(observed, rect) > threshold
                assert session.action_count == 0
                assert session.tick == 8
                assert (session.output / session.current_name).is_file()


def test_agent_plans_only_after_text_finishes(tmp_path):
    paths = (ROM, CORE, CHECKPOINTS / "face-npc.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoint required")

    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "face-npc.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()
        session.apply(Press("A"))
        assert GameRam(game).state().text_printing

        def stop():
            assert not GameRam(game).state().text_printing
            return True

        model = ScriptedModel(stop=stop)
        agent = Agent(model)
        agent.set_task("Read Granbull's line")
        assert isinstance(agent.act(session)[0].action, Stop)
        assert session.action_count == 1


def test_agent_waits_before_each_lombre_dialogue_turn(tmp_path):
    paths = (ROM, CORE, CHECKPOINTS / "face-lombre.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoint required")

    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "face-lombre.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()

        def stop():
            state = GameRam(game).state()
            assert not state.text_printing
            assert model.calls[-1][2].rgba == game.screenshot().rgba
            return state.mode == "field" and session.action_count > 0

        model = ScriptedModel(["A"], stop)
        agent = Agent(model)
        agent.set_task("Finish talking to Lombre")
        for _ in range(9):
            assert agent.act(session)[0].action == Press("A")
        assert isinstance(agent.act(session)[0].action, Stop)
        assert session.action_count == 9


def test_game_ram_rejects_another_rom(tmp_path):
    class OtherRom:
        rom = tmp_path / "other.gba"

    OtherRom.rom.write_bytes(b"not red rescue team")
    with pytest.raises(ValueError, match="Red Rescue Team ROM"):
        GameRam(OtherRom())


def test_textbox_text_reads_the_open_page():
    paths = (ROM, CORE, CHECKPOINTS / "npc-dialogue.ss0", CHECKPOINTS / "square.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoints required")
    with GbaEmulator(ROM, CORE) as game:
        ram = GameRam(game)
        restore(game, read_ss0(CHECKPOINTS / "npc-dialogue.ss0"))
        assert ram.textbox_text().startswith("Granbull:  I've given up")
        restore(game, read_ss0(CHECKPOINTS / "square.ss0"))
        assert ram.textbox_text() == ""


def test_observations_carry_the_game_state_before_and_after_a_move(tmp_path):
    if not all(path.is_file() for path in (ROM, CORE, CHECKPOINTS / "square.ss0")):
        pytest.skip("local ROM, core, and checkpoint required")
    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "square.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()
        observed = session.observe().game_state
        transition = session.apply(Press("DOWN"))
        assert transition.before.game_state == observed
        assert observed.tile is None and observed.direction == "south"
        x, y = observed.position
        after = transition.after.game_state
        assert after.position[0] == x and after.position[1] > y


def test_ground_characters_are_named_from_the_rom_and_filtered_to_the_screen(tmp_path):
    if not all(path.is_file() for path in (ROM, CORE, CHECKPOINTS / "square.ss0")):
        pytest.skip("local ROM, core, and checkpoint required")
    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "square.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()
        ram = GameRam(game)
        assert ram.species_name(235) == "Granbull" and ram.species_name(40) == "Wigglytuff"
        assert ram.npc_position(235) == (456, 352)
        assert len(ram.ground_characters()) > 10  # the whole square is loaded
        state = session.observe().game_state
        names = {character.name for character in state.characters}
        assert {"Granbull", "Lombre"} <= names and "Wigglytuff" not in names  # 260 px away
        assert "Quilava" not in names  # the player isn't listed as another character
        assert state.species == "Quilava"
