"""Checkpoint loading, observations and the game running in real time."""

import time

import pytest
from fakes import FRAME, Game

from emulator import Frame
from mystery_dungeon.actions import Press
from mystery_dungeon.session import TICK_SECONDS, Checkpoint, GameplaySession, RealtimeSession


def test_load_resets_counters_and_restores_save_ram(tmp_path):
    game = Game()
    session = GameplaySession(game, tmp_path)
    session.load(Checkpoint(b"start"))
    session.apply(Press("A"))
    session.load(Checkpoint(b"seed-state", b"seed-ram"))
    assert (game.state, game.ram) == (b"seed-state", b"seed-ram")
    assert (session.tick, session.action_count) == (1, 0)
    loaded = session.last_observation
    assert session.observe() == loaded  # nothing ran since the load


def test_failed_load_restores_the_previous_game(tmp_path):
    game = Game()
    session = GameplaySession(game, tmp_path)
    session.load(Checkpoint(b"start"))
    with pytest.raises(RuntimeError, match="invalid core state"):
        session.load(Checkpoint(b"bad", b"seed-ram"))
    assert (game.state, game.ram) == (b"start", b"current-ram")


def test_observe_after_idle_records_what_the_idle_ticks_did(tmp_path):
    game = Game()
    session = GameplaySession(game, tmp_path)
    before = session.observe()
    game.frame = Frame(16, 16, b"\xff" * len(FRAME.rgba))
    session.idle(3)
    after = session.observe()
    assert after.frame == game.frame and after.screenshot != before.screenshot
    assert session.observe() == after


def test_apply_needs_an_observation_first(tmp_path):
    session = GameplaySession(Game(), tmp_path)
    with pytest.raises(RuntimeError, match="no observation yet"):
        session.apply(Press("A"))


def test_the_game_runs_at_real_speed_between_observe_and_apply(tmp_path):
    session = GameplaySession(Game(), tmp_path)
    realtime = RealtimeSession(session)
    realtime.observe()
    tick = session.tick
    time.sleep(30 * TICK_SECONDS)
    realtime.apply(Press("A"))  # halts the game first
    idle = session.tick - tick - session.hold_ticks - session.release_ticks
    assert 10 <= idle <= 90  # about one tick per TICK_SECONDS, not thousands
    after = session.tick
    time.sleep(10 * TICK_SECONDS)
    assert session.tick == after  # nothing runs after the action until the next observe


def test_observing_again_keeps_one_idle_thread(tmp_path):
    session = GameplaySession(Game(), tmp_path)
    realtime = RealtimeSession(session)
    realtime.observe()
    realtime.observe()
    realtime.halt()
    tick = session.tick
    time.sleep(10 * TICK_SECONDS)
    assert session.tick == tick  # no thread from the first observe is left running


def test_a_tick_failure_is_raised_at_halt(tmp_path):
    game = Game()
    session = GameplaySession(game, tmp_path)
    realtime = RealtimeSession(session)
    realtime.observe()
    game.step = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("core crashed"))
    time.sleep(5 * TICK_SECONDS)
    with pytest.raises(RuntimeError, match="core crashed"):
        realtime.halt()
