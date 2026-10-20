"""The action space and how the session applies it."""

from dataclasses import dataclass

import pytest
from fakes import Game, start

from mystery_dungeon.actions import ACTIONS, GAME_ACTIONS, Done, Press, Wait, name, to_json
from mystery_dungeon.prompts import ACTION_DESCRIPTIONS


def test_names_round_trip_serialize_and_have_descriptions():
    assert all(name(action) == key for key, action in ACTIONS.items())
    assert ACTION_DESCRIPTIONS.keys() == ACTIONS.keys()
    assert ACTIONS.keys() - GAME_ACTIONS.keys() == {"DONE"}
    assert name(Done()) == "DONE" and Done() not in GAME_ACTIONS.values()
    assert to_json(Press("UP")) == {"type": "press", "button": "UP"}
    assert to_json(Wait()) == {"type": "wait"}


def test_session_presses_and_waits(tmp_path):
    game = Game()
    session = start(tmp_path, game)
    session.apply(Press("A"))
    session.apply(Wait())
    assert game.steps == [(8, ("A",)), (4, ()), (8, ()), (4, ())]
    assert session.action_count == 2


def test_session_rejects_unknown_actions(tmp_path):
    @dataclass(frozen=True)
    class Navigate:
        x: int
        y: int

    session = start(tmp_path)
    with pytest.raises(AssertionError):
        session.apply(Navigate(1, 2))
