"""Run short Pokémon Square tasks with hidden game-state grading."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from emulator import GbaEmulator
from mystery_dungeon.agent import Agent
from mystery_dungeon.models import CONFIG, Model, load_model, read_models
from mystery_dungeon.models.server import ModelServer
from mystery_dungeon.ram_state import GameRam
from mystery_dungeon.run_log import FORMAT_VERSION, RunLog
from mystery_dungeon.session import (HOLD_FRAMES, RELEASE_FRAMES, Checkpoint, GameplaySession,
                                     read_ss0)


GAME = tomllib.loads(CONFIG.read_text())["game"]
ROM, CORE, CHECKPOINTS = (CONFIG.parent / GAME[key] for key in ("rom", "core", "checkpoints"))
GRANBULL, LOMBRE, WIGGLYTUFF = 235, 296, 40  # species ids
WHISCASH_POND = 3  # ground map beyond the square's north exit


@dataclass(frozen=True)
class Case:
    id: str
    start: str
    task: str
    criterion: str
    limit: int


CASES = (
    Case("square_down", "square", "Move one tile down", "down", 5),
    Case("square_north_exit", "square",
         "Walk out of Pokémon Square through the north exit", "north_exit", 26),
    Case("square_approach_granbull", "square",
         "Stand just above the purple Pokémon on the left side of the square", "approach", 12),
    Case("square_talk_granbull", "square",
         "Walk to Granbull in the square and start a conversation", "granbull", 16),
    Case("face_npc_talk", "face-lombre",
         "Open dialogue with Lombre, the green Pokémon immediately to your right. "
         "Stop when Lombre's first dialogue box appears; do not advance or finish the conversation",
         "lombre", 5),
    Case("face_lombre_finish", "face-lombre",
         "Talk to Lombre, the green Pokémon immediately to your right, until the whole "
         "conversation ends. Stop when the dialogue box closes and you are back in the square",
         "finish_lombre", 16),
    Case("face_npc_talk_overlap", "face-npc",
         "Open dialogue with Granbull directly in front of you. "
         "Stop when its first dialogue box appears; do not advance or finish the conversation",
         "granbull", 5),
    Case("dialogue_finish", "npc-dialogue", "Finish the conversation with Granbull", "finish", 18),
    Case("face_npc_finish", "face-npc", "Talk to Granbull until the conversation ends", "finish", 22),
    Case("shop_talk", "shop-interact", "Talk to the shopkeeper in front of you", "wigglytuff", 5),
    Case("shop_buy_friend_area", "shop-interact",
         "Open the Friend Areas list at Wigglytuff's shop and choose one to buy", "buy_reply", 24),
    Case("shop_check_pidgey", "shop-interact",
         "Choose Check at Wigglytuff's shop, then select Pidgey from the Pokémon list",
         "check_pidgey", 30),
)

CHECKPOINT_NAMES = tuple(dict.fromkeys(case.start for case in CASES))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Grader:
    """Judge one case's criterion from general game facts, independently of the agent."""

    def __init__(self, case: Case, ram: GameRam):
        self.case = case
        self.ram = ram
        position = ram.player_position()
        if position is None:
            raise ValueError(f"player position unavailable in {case.start} checkpoint")
        self.initial_x, self.initial_y = position
        self.saw_granbull = self.talking(GRANBULL)
        self.saw_lombre = self.talking(LOMBRE)

    def talking(self, species: int) -> bool:
        return self.ram.textbox_type() == 1 and self.ram.speaker() == species

    def reached(self) -> bool:
        position = self.ram.player_position()
        if position is None:
            return False
        x, y = position
        closed = self.ram.textbox_type() in (None, 0)
        self.saw_granbull |= self.talking(GRANBULL)
        self.saw_lombre |= self.talking(LOMBRE)
        match self.case.criterion:
            case "down":
                return x == self.initial_x and y >= self.initial_y + 16
            case "north_exit":
                return self.ram.ground_map() == WHISCASH_POND
            case "approach":
                npc = self.ram.npc_position(GRANBULL)
                return (npc is not None and abs(x - npc[0]) <= 12 and
                        8 <= npc[1] - y <= 24 and closed)
            case "granbull":
                return self.talking(GRANBULL)
            case "lombre":
                return self.talking(LOMBRE)
            case "finish":
                return self.saw_granbull and closed
            case "finish_lombre":
                return self.saw_lombre and closed
            case "wigglytuff":
                return self.ram.textbox_type() == 4 and self.ram.speaker() == WIGGLYTUFF
            case "buy_reply":
                # The checkpoint has no money, so buying any Friend Area ends at this reply.
                return "You don't have any money." in self.ram.textbox_text()
            case "check_pidgey":
                return "Pidgey's Friend Area" in self.ram.textbox_text()
            case criterion:
                raise ValueError(f"unknown criterion: {criterion}")


def build_fixtures(game: GbaEmulator, output: Path) -> dict[str, Checkpoint]:
    checkpoints = {name: read_ss0(CHECKPOINTS / f"{name}.ss0") for name in CHECKPOINT_NAMES}
    folder = output / "references"
    folder.mkdir()
    session = GameplaySession(game, folder)  # saves each checkpoint's frame under frames/
    for name, checkpoint in checkpoints.items():
        (folder / f"{name}.state").write_bytes(checkpoint.state)
        (folder / f"{name}.sav").write_bytes(checkpoint.save_ram)
        session.load(checkpoint)
        for case in (case for case in CASES if case.start == name):
            if Grader(case, session.ram).reached():
                raise ValueError(f"{case.id} already passes at its checkpoint")
    return checkpoints


def run_case(game: GbaEmulator, case: Case, model: Model,
             checkpoints: dict[str, Checkpoint], output: Path,
             knowledge: str = "", model_name: str = "injected") -> dict:
    output.mkdir(parents=True, exist_ok=False)
    log = RunLog(output, model_name,
                 {"kind": "benchmark_case", "case": case.id, "start": case.start,
                  "criterion": case.criterion, "limit": case.limit,
                  "hold_frames": HOLD_FRAMES, "release_frames": RELEASE_FRAMES},
                 knowledge)
    watched = log.watch(model)
    agent = Agent(watched, watched, knowledge)
    agent.set_instructions(case.task)
    session = GameplaySession(game, output)
    session.load(checkpoints[case.start])
    if session.ram is None:
        raise ValueError("benchmark grading needs a real emulator")
    grader = Grader(case, session.ram)
    started = time.monotonic()
    reason = "limit"
    error = None
    turns = 0
    reached_at = None
    while True:
        success = grader.reached()
        if case.criterion in ("finish", "finish_lombre") and not success:
            reached_at = None  # A reopened the conversation after it had ended.
        elif success and reached_at is None:
            reached_at = turns
        if session.action_count >= case.limit:
            reason = "limit"
            break
        turn, transition = agent.take_turn(session)
        log.record_turn(turn, transition)
        if turn.status == "model_error":
            reason, error = "model_error", turn.error
            break
        turns += 1
        if turn.status == "completion":
            reason = "recognized" if reached_at is not None else "false_completion"
            break
    final = session.current
    final.save_png(output / "frames/final.png")
    (output / "states").mkdir(exist_ok=True)
    (output / "states/final.state").write_bytes(game.save_state())
    (output / "states/final.sav").write_bytes(game.save_ram())
    public_reason = ("completion_claimed" if reason in ("recognized", "false_completion")
                     else reason)
    log.record_run_stop(public_reason, session.current_name)
    reached = reached_at is not None
    result = {"case": case.id, "task": case.task, "reached": reached,
              "recognized": reason == "recognized", "missed": reached and reason != "recognized",
              "false_completion": reason == "false_completion",
              "reached_at": reached_at, "stop_reason": reason,
              "actions": session.action_count, "turns": turns,
              "seconds": round(time.monotonic() - started, 2), "error": error}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def report(results: list[dict], output: Path) -> None:
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Game task benchmark", "",
             "| Case | Reached | Recognized | Missed | False completion | Actions | Seconds |",
             "|---|:---:|:---:|:---:|:---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['case']} | {row['reached']} | {row['recognized']} | "
                     f"{row['missed']} | {row['false_completion']} | {row['actions']} | {row['seconds']} |")
    lines += ["", "The hidden grader measures reach independently from model completion.",
              "Each turn makes separate completion and action requests, so latency is not "
              "comparable with earlier benchmark reports.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, default=ROM)
    parser.add_argument("--core", type=Path, default=CORE)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/benchmarks") / datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
                        help="run directory (default: timestamped directory in runs/benchmarks)")
    models = read_models()
    parser.add_argument("--model", required=True, choices=models,
                        help="a [models.NAME] table in config.toml")
    parser.add_argument("--knowledge", type=Path, help="text the agent starts out knowing")
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    args = parser.parse_args()
    selected = [case for case in CASES if args.case is None or case.id in args.case]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases").mkdir()
    knowledge = args.knowledge.read_text() if args.knowledge else ""
    entry = models[args.model]
    files = (args.rom, args.core, *(CHECKPOINTS / f"{name}.ss0" for name in CHECKPOINT_NAMES),
             *(CONFIG.parent / path for path in entry.files),
             *((args.knowledge,) if args.knowledge else ()))
    hashes = {str(path): sha256(path) for path in files}
    (args.output / "config.json").write_text(json.dumps({
        "format_version": FORMAT_VERSION, "layout": "benchmark", "grader": "pmd-red-ram-v1",
        "model": asdict(entry), "cases": [asdict(c) for c in selected],
        "hashes": hashes, "knowledge": str(args.knowledge) if args.knowledge else None,
        "hold_frames": HOLD_FRAMES, "release_frames": RELEASE_FRAMES,
    }, indent=2) + "\n")
    with GbaEmulator(args.rom, args.core) as game, closing(ModelServer(args.output)) as server:
        url = server.start(entry)
        checkpoints = build_fixtures(game, args.output)
        results = []
        for case in selected:
            print(case.id, flush=True)
            result = run_case(game, case, load_model(entry.spec, url), checkpoints,
                              args.output / "cases" / case.id, knowledge, args.model)
            results.append(result)
            report(results, args.output)
            print(f"  {result['stop_reason']} after {result['actions']} actions", flush=True)


if __name__ == "__main__":
    main()
