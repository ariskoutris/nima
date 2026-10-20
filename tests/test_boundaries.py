"""The agent and models see frames and buttons only."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "mystery_dungeon"
FORBIDDEN = {"mystery_dungeon.ram_state", "mystery_dungeon.benchmark"}


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_agent_and_models_never_import_game_state():
    for path in (PACKAGE / "agent.py", PACKAGE / "prompts.py", PACKAGE / "model.py"):
        assert not imported_modules(path) & FORBIDDEN, path
