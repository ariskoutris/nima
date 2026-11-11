"""The action space and how the session applies it."""

from dataclasses import dataclass

import pytest
from fakes import Game, start

from nima.actions import (
    ACTION_NAMES,
    GAME_ACTIONS,
    Press,
    Stop,
    Wait,
    name,
    to_json,
)
from nima.prompts import ACTION_DESCRIPTIONS


def test_names_round_trip_serialize_and_have_descriptions():
    assert all(name(action) == key for key, action in GAME_ACTIONS.items())
    assert tuple(ACTION_DESCRIPTIONS) == ACTION_NAMES
    assert set(ACTION_NAMES) - GAME_ACTIONS.keys() == {
        "STOP_COMPLETE", "STOP_STUCK", "STOP_INVALID"}
    assert name(Stop("stuck")) == "STOP_STUCK"
    assert to_json(Stop("stuck")) == {"type": "stop", "outcome": "stuck"}
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
