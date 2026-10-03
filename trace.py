"""Replay GBA button batches and record the resulting visible frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from emulator import DEFAULT_CORE, GbaEmulator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(rom: Path, actions: Path, output: Path, core: Path = DEFAULT_CORE,
           save_ram: Path | None = None, start_state: Path | None = None) -> None:
    if save_ram is not None and start_state is not None:
        raise ValueError("use either a battery save or an emulator state")
    output.mkdir(parents=True, exist_ok=True)
    steps = json.loads(actions.read_text())
    if not isinstance(steps, list):
        raise ValueError("actions file must contain a JSON list")

    metadata = {
        "rom_sha256": sha256(rom),
        "core_sha256": sha256(core),
        "actions": str(actions),
    }
    if save_ram is not None:
        metadata["save_ram_sha256"] = sha256(save_ram)
    if start_state is not None:
        metadata["start_state_sha256"] = sha256(start_state)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    frame_number = 0
    with GbaEmulator(rom, core) as emulator, (output / "frames.jsonl").open("w") as log:
        if save_ram is not None:
            emulator.load_save_ram(save_ram.read_bytes())
        if start_state is not None:
            emulator.load_state(start_state.read_bytes())
        for index, action in enumerate(steps):
            frames = action["frames"]
            buttons = tuple(action.get("buttons", ()))
            emulator.step(frames, buttons)
            frame_number += frames
            screenshot = emulator.screenshot()
            filename = f"{index:03d}.png"
            screenshot.save_png(output / filename)
            row = {
                "frame": frame_number,
                "buttons": buttons,
                "duration_frames": frames,
                "screenshot": filename,
                "rgba_sha256": hashlib.sha256(screenshot.rgba).hexdigest(),
            }
            if "note" in action:
                row["note"] = action["note"]
            log.write(json.dumps(row) + "\n")
        (output / "last.state").write_bytes(emulator.save_state())
        (output / "last.sav").write_bytes(emulator.save_ram())
    print(f"Recorded {len(steps)} actions through frame {frame_number} in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("actions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    starting_point = parser.add_mutually_exclusive_group()
    starting_point.add_argument("--save-ram", type=Path, help="in-game .sav file")
    starting_point.add_argument("--start-state", type=Path, help="mGBA Libretro checkpoint")
    args = parser.parse_args()
    record(args.rom, args.actions, args.output, args.core, args.save_ram,
           args.start_state)


if __name__ == "__main__":
    main()
