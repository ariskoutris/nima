"""RAM observations against the checked-in Red Rescue Team checkpoints."""

import pytest

from emulator import GbaEmulator
from mystery_dungeon.actions import Press
from mystery_dungeon.agent import Agent
from fakes import ScriptedModel, press
from mystery_dungeon.benchmark import CHECKPOINTS, CORE, ROM
from mystery_dungeon.ram_state import GameRam
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
        waited, ready = session.wait_for_text()
        assert ready and waited > 0
        assert session.action_count == actions
        assert GameRam(game).state().mode == ready_mode
        assert not GameRam(game).state().text_printing
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
                assert session.frame_count == 8
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

        def complete():
            assert not GameRam(game).state().text_printing
            return True

        model = ScriptedModel(complete=complete)
        agent = Agent(model, model)
        agent.set_instructions("Read Granbull's line")
        assert agent.take_turn(session)[0].status == "completion"
        assert session.action_count == 1


def test_agent_waits_before_each_lombre_dialogue_turn(tmp_path):
    paths = (ROM, CORE, CHECKPOINTS / "face-lombre.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoint required")

    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "face-lombre.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()

        def complete():
            state = GameRam(game).state()
            assert not state.text_printing
            assert model.calls[-1][2].rgba == game.screenshot().rgba
            return state.mode == "field" and session.action_count > 0

        model = ScriptedModel(["A"], complete)
        agent = Agent(model, model)
        agent.set_instructions("Finish talking to Lombre")
        for _ in range(9):
            assert agent.take_turn(session)[0].status == "action"
        assert agent.take_turn(session)[0].status == "completion"
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
