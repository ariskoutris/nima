"""The real emulator runs quietly."""

import pytest

from emulator import GbaEmulator
from nima.benchmark import CHECKPOINTS, CORE, ROM
from nima.session import GameplaySession, read_checkpoint


def test_core_log_messages_are_not_printed(tmp_path, capfd):
    checkpoint = CHECKPOINTS / "square.ss0"
    if not (ROM.is_file() and CORE.is_file() and checkpoint.is_file()):
        pytest.skip("local ROM, core and checkpoint required")
    with GbaEmulator(ROM, CORE) as game:
        GameplaySession(game, tmp_path).load(read_checkpoint(checkpoint))
    assert "%s" not in capfd.readouterr().err
