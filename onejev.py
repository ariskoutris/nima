"""Replay a checkpoint with OneJev choosing buttons from screenshots."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

from qev import Choice, Client

from emulator import GbaEmulator


OPTIONS = {
    "UP": "press up on the directional pad",
    "DOWN": "press down on the directional pad",
    "LEFT": "press left on the directional pad",
    "RIGHT": "press right on the directional pad",
    "A": "confirm, interact, or advance text",
    "B": "cancel or go back",
    "START": "press the Start button",
    "SELECT": "press the Select button",
    "L": "press the left shoulder button",
    "R": "press the right shoulder button",
    "WAIT": "press nothing and let the game advance",
}


def choose(client: Client, screenshot: Path, goal: str, recent: list[str]) -> dict:
    image = base64.b64encode(screenshot.read_bytes()).decode("ascii")
    started = time.perf_counter()
    response = client.system_one(
        state={"goal": goal, "recent_buttons": recent[-4:], "screen": "<image:1>"},
        media=[{"type": "image", "data": "data:image/png;base64," + image}],
        questions={"next": Choice("Which gamepad action should be taken next to make progress?", OPTIONS)},
    )
    answer = response.answers["next"]
    if answer.choice not in OPTIONS:
        raise RuntimeError(f"OneJev returned an unknown action: {answer.choice!r}")
    return {
        "choice": answer.choice,
        "probabilities": answer.probabilities,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("--state", type=Path, default=Path(".local/baselines/post_continue.state"))
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--release-frames", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.hold_frames, args.release_frames) < 1:
        parser.error("steps and frame counts must be positive")
    args.output.mkdir(parents=True)
    client = Client(args.url)
    recent: list[str] = []
    with GbaEmulator(args.rom) as emulator, (args.output / "decisions.jsonl").open("w") as log:
        emulator.load_state(args.state.read_bytes())
        emulator.step()
        for index in range(args.steps):
            screenshot = args.output / f"{index:03d}.png"
            emulator.screenshot().save_png(screenshot)
            decision = choose(client, screenshot, args.goal, recent)
            button = decision["choice"]
            emulator.step(args.hold_frames, () if button == "WAIT" else (button,))
            emulator.step(args.release_frames)
            row = {"screenshot": screenshot.name, "goal": args.goal, **decision}
            log.write(json.dumps(row) + "\n")
            log.flush()
            recent.append(button)
            print(f"{index}: {button} ({decision['latency_ms']} ms)")
        emulator.screenshot().save_png(args.output / "final.png")
        (args.output / "final.state").write_bytes(emulator.save_state())


if __name__ == "__main__":
    main()
