"""Replay a checkpoint toward one manually supplied short-term goal."""

from __future__ import annotations

import argparse
from pathlib import Path

from emulator import GbaEmulator
from mystery_dungeon.agent import Agent, load_decision_model
from mystery_dungeon.session import GameplaySession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision-model", default="mystery_dungeon.decision_models.onejev:OneJevDecisionModel")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--history-frames", type=int, default=1)
    parser.add_argument("--completion-votes", type=int, default=1)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--release-frames", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.steps, args.hold_frames, args.release_frames, args.completion_votes) < 1 \
            or args.history_frames < 0:
        parser.error("steps and frame counts must be positive; history frames may be zero")
    args.output.mkdir(parents=True, exist_ok=False)
    agent = Agent(load_decision_model(args.decision_model), args.output, args.decision_model,
                  args.history_frames, args.completion_votes,
                  args.context.read_text() if args.context else "",
                  {"kind": "play", "rom": str(args.rom), "core": str(args.core),
                   "initial_state": str(args.state), "hold_frames": args.hold_frames,
                   "release_frames": args.release_frames, "step_limit": args.steps})
    agent.set_goal(args.goal)
    with GbaEmulator(args.rom, args.core) as emulator:
        emulator.load_state(args.state.read_bytes())
        emulator.step()
        session = GameplaySession(emulator, agent, args.output,
                                  args.hold_frames, args.release_frames)
        session.frame_count = 1
        session.capture()
        for index in range(args.steps):
            decision, stop = session.decide()
            if stop:
                session.stop(decision)
                print(f"{index}: goal complete")
                break
            button = decision.choice
            session.apply(button, "agent", decision)
            print(f"{index}: {button} ({decision.latency_ms} ms)")
        else:
            agent.record({"event": "run_stop", "reason": "step_limit",
                          "screenshot": session.current_name})
        session.current.save_png(args.output / "frames/final.png")
        (args.output / "states").mkdir(exist_ok=True)
        (args.output / "states/final.state").write_bytes(emulator.save_state())


if __name__ == "__main__":
    main()
