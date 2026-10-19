"""Replay a checkpoint toward one task given on the command line."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

from emulator import GbaEmulator
from mystery_dungeon.actions import name
from mystery_dungeon.agent import Agent
from mystery_dungeon.models import load_model, read_models
from mystery_dungeon.models.server import ModelServer
from mystery_dungeon.run_log import RunLog
from mystery_dungeon.session import HOLD_FRAMES, RELEASE_FRAMES, GameplaySession, read_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    models = read_models()
    parser.add_argument("--model", default=next(iter(models)), choices=models,
                        help="a [models.NAME] table in config.toml (default: the first)")
    parser.add_argument("--knowledge", type=Path, help="text the agent starts out knowing")
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--hold-frames", type=int, default=HOLD_FRAMES)
    parser.add_argument("--release-frames", type=int, default=RELEASE_FRAMES)
    args = parser.parse_args(argv)
    if min(args.turns, args.hold_frames, args.release_frames) < 1:
        parser.error("turns and frame counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    knowledge = args.knowledge.read_text() if args.knowledge else ""
    log = RunLog(args.output, args.model,
                 {"kind": "play", "rom": str(args.rom), "core": str(args.core),
                  "initial_state": str(args.state), "hold_frames": args.hold_frames,
                  "release_frames": args.release_frames, "turn_limit": args.turns},
                 knowledge)
    entry = models[args.model]
    with closing(ModelServer(args.output)) as server, GbaEmulator(args.rom, args.core) as emulator:
        model = log.watch(load_model(entry.spec, server.start(entry)))
        agent = Agent(model, model, knowledge)
        agent.set_instructions(args.task)
        session = GameplaySession(emulator, args.output, args.hold_frames, args.release_frames)
        session.load(read_checkpoint(args.state))
        for index in range(args.turns):
            turn, transition = agent.take_turn(session)
            log.record_turn(turn, transition)
            if turn.status == "model_error":
                raise RuntimeError(turn.error)
            if turn.status == "completion":
                print(f"{index}: task complete")
                break
            print(f"{index}: {name(turn.action)}")
        else:
            log.record_run_stop("turn_limit", session.current_name)
        session.current.save_png(args.output / "frames/final.png")
        (args.output / "states").mkdir(exist_ok=True)
        (args.output / "states/final.state").write_bytes(emulator.save_state())


if __name__ == "__main__":
    main()
