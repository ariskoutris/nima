"""Replay a checkpoint with any classifier that scores buttons from a screenshot.

A classifier is a callable (screenshot_path, goal, recent_buttons, options)
returning one finite score per option. Higher scores are better.
Select an importable callable with --classifier module:callable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Callable, Mapping

from emulator import DEFAULT_CORE, GbaEmulator


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

Classifier = Callable[[Path, str, tuple[str, ...], Mapping[str, str]], Mapping[str, float]]


def load_classifier(spec: str) -> Classifier:
    module, separator, name = spec.partition(":")
    if not (module and separator and name):
        raise ValueError("classifier must be MODULE:CALLABLE")
    classifier = getattr(importlib.import_module(module), name)
    if not callable(classifier):
        raise TypeError(f"{spec} is not callable")
    return classifier


def choose(classifier: Classifier, screenshot: Path, goal: str, recent: list[str]) -> dict:
    started = time.perf_counter()
    scores = dict(classifier(screenshot, goal, tuple(recent[-4:]), OPTIONS.copy()))
    if scores.keys() != OPTIONS.keys() or any(
        not isinstance(score, (int, float)) or not math.isfinite(score)
        for score in scores.values()
    ):
        raise ValueError("classifier must return one finite score for every button option")
    return {
        "choice": max(scores, key=scores.get),
        "scores": scores,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


def main(argv: list[str] | None = None, default_classifier: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--state", type=Path, default=Path(".local/baselines/post_continue.state"))
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--classifier", required=default_classifier is None,
                        default=default_classifier, metavar="MODULE:CALLABLE")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--release-frames", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.steps, args.hold_frames, args.release_frames) < 1:
        parser.error("steps and frame counts must be positive")
    classifier = load_classifier(args.classifier)
    args.output.mkdir(parents=True)
    recent: list[str] = []
    with GbaEmulator(args.rom, args.core) as emulator, (args.output / "decisions.jsonl").open("w") as log:
        emulator.load_state(args.state.read_bytes())
        emulator.step()
        for index in range(args.steps):
            screenshot = args.output / f"{index:03d}.png"
            emulator.screenshot().save_png(screenshot)
            decision = choose(classifier, screenshot, args.goal, recent)
            button = decision["choice"]
            emulator.step(args.hold_frames, () if button == "WAIT" else (button,))
            emulator.step(args.release_frames)
            row = {"screenshot": screenshot.name, "goal": args.goal,
                   "classifier": args.classifier, **decision}
            log.write(json.dumps(row) + "\n")
            log.flush()
            recent.append(button)
            print(f"{index}: {button} ({decision['latency_ms']} ms)")
        emulator.screenshot().save_png(args.output / "final.png")
        (args.output / "final.state").write_bytes(emulator.save_state())


if __name__ == "__main__":
    main()
