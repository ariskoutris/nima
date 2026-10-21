"""Checkpoint loading and observations."""

import pytest
from fakes import Game

from mystery_dungeon.actions import Press
from mystery_dungeon.session import Checkpoint, GameplaySession


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


def test_apply_needs_an_observation_first(tmp_path):
    session = GameplaySession(Game(), tmp_path)
    with pytest.raises(RuntimeError, match="no observation yet"):
        session.apply(Press("A"))
