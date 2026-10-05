"""Capture one-action probes from scripted, verified routes without model input."""

import json
from pathlib import Path

from emulator import GbaEmulator
from mystery_dungeon.benchmark import CORE, ROM, Seed, press, restore


ROOT = Path("runs/benchmark-20261004-4b-plain/references")
OUT = Path("runs/one-step-diagnostics/heldout")
CASES = {
    "room_after_down1": ("inside", ("DOWN",), "Move one tile down toward the doorway below", "DOWN"),
    "room_after_down3": ("inside", ("DOWN",) * 3, "Move one tile down toward the doorway below", "DOWN"),
    "room_goal_after3": ("inside", ("DOWN",) * 3, "Leave the base", "DOWN"),
    "outside_right1": ("outside", ("RIGHT",), "Move one tile right toward the wooden bridge", "RIGHT"),
    "outside_right3": ("outside", ("RIGHT",) * 3, "Move one tile right toward the wooden bridge", "RIGHT"),
    "outside_right6": ("outside", ("RIGHT",) * 6, "Move one tile right toward the wooden bridge", "RIGHT"),
    "bridge_up1": ("bridge", ("UP",), "Move one tile up toward the wooded clearing", "UP"),
    "outside_npc": ("outside", (), "Move one tile down toward the green Pokemon below", "DOWN"),
    "outside_mailbox": ("outside", (), "Move one tile left toward the blue mailbox", "LEFT"),
    "open_menu": ("outside", (), "Open the game menu", "B"),
    "mailbox_dialogue": ("outside", ("LEFT", "A"), "Advance the visible message", "A"),
    "npc_dialogue": ("outside", ("DOWN", "A"), "Advance the visible dialogue", "A"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with GbaEmulator(ROM, CORE) as game:
        for case, (start, route, goal, expected) in CASES.items():
            seed = Seed((ROOT / f"{start}.state").read_bytes(), (ROOT / f"{start}.sav").read_bytes())
            restore(game, seed)
            for button in route:
                press(game, button)
            game.screenshot().save_png(OUT / f"{case}.png")
            manifest[case] = {"image": f"{case}.png", "goal": goal, "expected": expected,
                              "start": start, "route": route}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
