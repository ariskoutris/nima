"""Game observation and control; the configured game paths."""

import tomllib
from pathlib import Path

CONFIG = Path(__file__).parents[1] / "config.toml"
_GAME = tomllib.loads(CONFIG.read_text())["game"]
ROM, CORE, STATE, CHECKPOINTS = (CONFIG.parent / _GAME[key]
                                 for key in ("rom", "core", "state", "checkpoints"))
