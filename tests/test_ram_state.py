"""RAM observations against the checked-in Red Rescue Team checkpoints."""

import pytest

from emulator import GbaEmulator
from mystery_dungeon.agent import Agent, DecisionScores
from mystery_dungeon.benchmark import CHECKPOINTS, CORE, ROM, press, read_ss0, restore
from mystery_dungeon.ram_state import read_ram_state
from mystery_dungeon.session import GameplaySession


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
        assert read_ram_state(game).mode == "field"
        session.apply("A")
        initial = read_ram_state(game)
        assert initial.mode == "dialogue" and initial.text_printing
        partial = session.current.rgba
        actions = session.action_count
        waited, ready = session.wait_for_text()
        assert ready and waited > 0
        assert session.action_count == actions
        assert read_ram_state(game).mode == ready_mode
        assert not read_ram_state(game).text_printing
        assert session.current.rgba != partial


def test_field_menu_and_dungeon_modes():
    paths = (ROM, CORE, CHECKPOINTS / "square.ss0", CHECKPOINTS / "normal-maze.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoints required")
    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "square.ss0"))
        field = read_ram_state(game)
        assert field.mode == "field" and field.player_position is not None
        game.step(8, ("B",))
        game.step(4)
        assert read_ram_state(game).mode == "menu"
        restore(game, read_ss0(CHECKPOINTS / "normal-maze.ss0"))
        assert read_ram_state(game).mode == "dungeon"


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
        while read_ram_state(game).text_printing:
            game.step(4)
        game.step(12)
        popup = game.save_state()

        press(game, "DOWN")
        press(game, "A")
        while read_ram_state(game).text_printing:
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
                assert read_ram_state(game).mode == "menu"
                observed = session.current_frame()
                assert light_pixels(observed, rect) > threshold
                assert session.action_count == 0
                assert session.frame_count == 8
                assert (session.output / session.current_name).is_file()


def test_agent_scores_only_after_text_finishes(tmp_path):
    paths = (ROM, CORE, CHECKPOINTS / "face-npc.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoint required")

    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "face-npc.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()
        session.apply("A")
        assert read_ram_state(game).text_printing

        class Model:
            def score(self, goal, guidance, frame, history, options):
                assert not read_ram_state(game).text_printing
                return DecisionScores({key: 0.0 for key in options},
                                      {"complete": 1.0, "incomplete": 0.0, "uncertain": 0.0})

        agent = Agent(Model(), tmp_path, "test:Model")
        agent.set_goal("Read Granbull's line")
        assert agent.step(session).status == "completed"
        assert session.action_count == 1


def test_agent_waits_before_each_lombre_dialogue_decision(tmp_path):
    paths = (ROM, CORE, CHECKPOINTS / "face-lombre.ss0")
    if not all(path.is_file() for path in paths):
        pytest.skip("local ROM, core, and checkpoint required")

    with GbaEmulator(ROM, CORE) as game:
        restore(game, read_ss0(CHECKPOINTS / "face-lombre.ss0"))
        session = GameplaySession(game, tmp_path)
        session.capture()

        class Model:
            def score(self, goal, guidance, frame, history, options):
                state = read_ram_state(game)
                assert not state.text_printing
                assert frame.rgba == game.screenshot().rgba
                done = state.mode == "field" and session.action_count > 0
                return DecisionScores({key: float(key == "A") for key in options},
                                      {"complete": float(done), "incomplete": float(not done),
                                       "uncertain": 0.0})

        agent = Agent(Model(), tmp_path, "test:Model")
        agent.set_goal("Finish talking to Lombre")
        for _ in range(9):
            assert agent.step(session).status == "acted"
        assert agent.step(session).status == "completed"
        assert session.action_count == 9
