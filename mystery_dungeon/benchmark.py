"""Run short Pokémon Square tasks with hidden game-state grading."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from emulator import GbaEmulator
from mystery_dungeon.agent import Agent
from mystery_dungeon.models import Model, load_model
from mystery_dungeon.run_log import RunLog
from mystery_dungeon.session import Checkpoint, GameplaySession, read_ss0, restore


ROM = Path("data/game/roms/red-rescue-team.gba")
CORE = Path("data/engines/cores/mgba_libretro.dylib")
CHECKPOINTS = Path("data/game/checkpoints")
ROM_SHA1 = "9f4cfc5b5f4859d17169a485462e977c7aac2b89"
HOLD, RELEASE = 8, 4


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
    Case("shop_friend_areas", "shop-interact", "Open the Friend Areas list at the shop", "shop_menu", 22),
    Case("shop_check_pidgey", "shop-interact",
         "Choose Check at Wigglytuff's shop, then select Pidgey from the Pokémon list",
         "check_pidgey", 30),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Grader:
    """Read fields documented by https://github.com/pret/pmd-red for ROM_SHA1."""

    def __init__(self, case: Case, game: GbaEmulator):
        self.game = game
        position = self.player_position()
        if position is None:
            raise ValueError(f"player position unavailable in {case.start} checkpoint")
        self.initial_x, self.initial_y = position
        self.saw_granbull = self.dialogue() == (1, 235)
        self.saw_lombre = self.dialogue() == (1, 296)

    def value(self, address: int, fmt: str) -> int:
        return struct.unpack(fmt, self.game.read_memory(address, struct.calcsize(fmt)))[0]

    def player_position(self) -> tuple[int, int] | None:
        # gGroundLives -> GroundLive[0].unk144 (PixelPos, s24.8).
        lives = self.value(0x03001B84, "<I")
        if not 0x02000000 <= lives < 0x02040000:
            return None
        x, y = struct.unpack("<ii", self.game.read_memory(lives + 0x144, 8))
        return x // 256, y // 256

    def ground_map_id(self) -> int | None:
        # gGroundMapAction -> groundMapId; 3 is Whiscash Pond beyond the north exit.
        action = self.value(0x03001B6C, "<I")
        return self.value(action + 0xE4, "<h") if 0x02000000 <= action < 0x02040000 else None

    def granbull_position(self) -> tuple[int, int] | None:
        lives = self.value(0x03001B84, "<I")
        if not 0x02000000 <= lives < 0x02040000:
            return None
        for index in range(24):
            entry = lives + index * 0x1F0
            if (self.value(entry + 2, "<h") != -1 and
                    self.value(entry + 8, "<h") == 235):
                x, y = struct.unpack("<ii", self.game.read_memory(entry + 0x144, 8))
                return x // 256, y // 256
        return None

    def dialogue(self) -> tuple[int, int]:
        # sTextbox -> Textbox.type and second portrait's speciesID.
        textbox = self.value(0x03001B64, "<I")
        if not 0x02000000 <= textbox < 0x02040000:
            return 0, -1
        return (self.value(textbox, "<I"),
                self.value(textbox + 0x43C + 0x24 + 2, "<h"))

    def shop_state(self) -> int | None:
        # WigglytuffShop3Work pointer -> state; states 9/10 show Friend Areas.
        shop = self.value(0x0203B290, "<I")
        return self.value(shop + 4, "<i") if 0x02000000 <= shop < 0x02040000 else None

    def checked_species(self) -> int | None:
        # WigglytuffShop3Work.state 24 displays chosenSpecies at +0x12.
        shop = self.value(0x0203B290, "<I")
        if 0x02000000 <= shop < 0x02040000 and self.value(shop + 4, "<i") == 24:
            return self.value(shop + 0x12, "<h")
        return None

    def observe(self, criterion: str) -> tuple[bool, list[str]]:
        position = self.player_position()
        if position is None:
            return False, []
        x, y = position
        dialogue, speaker = self.dialogue()
        granbull = dialogue == 1 and speaker == 235
        lombre = dialogue == 1 and speaker == 296
        if granbull:
            self.saw_granbull = True
        if lombre:
            self.saw_lombre = True
        npc = self.granbull_position() if criterion == "approach" else None
        success = {
            "down": x == self.initial_x and y >= self.initial_y + 16,
            "north_exit": self.ground_map_id() == 3 if criterion == "north_exit" else False,
            "approach": (npc is not None and abs(x - npc[0]) <= 12 and
                         8 <= npc[1] - y <= 24 and dialogue == 0),
            "granbull": granbull,
            "lombre": lombre,
            "finish": self.saw_granbull and dialogue == 0,
            "finish_lombre": self.saw_lombre and dialogue == 0,
            "wigglytuff": dialogue == 4 and speaker == 40,
            "shop_menu": self.shop_state() in (9, 10) if criterion == "shop_menu" else False,
            "check_pidgey": self.checked_species() == 16 if criterion == "check_pidgey" else False,
        }[criterion]
        return success, []


def build_fixtures(game: GbaEmulator, output: Path) -> dict[str, Checkpoint]:
    if hashlib.sha1(game.rom.read_bytes()).hexdigest() != ROM_SHA1:
        raise ValueError("town RAM offsets require the matching Red Rescue Team ROM")
    checkpoints = {name: read_ss0(CHECKPOINTS / f"{name}.ss0")
             for name in ("square", "face-lombre", "face-npc", "npc-dialogue", "shop-interact")}
    folder = output / "references"
    folder.mkdir()
    for name, checkpoint in checkpoints.items():
        (folder / f"{name}.state").write_bytes(checkpoint.state)
        (folder / f"{name}.sav").write_bytes(checkpoint.save_ram)
        restore(game, checkpoint)
        game.screenshot().save_png(folder / f"{name}.png")
        for case in (case for case in CASES if case.start == name):
            grader = Grader(case, game)
            if grader.observe(case.criterion)[0]:
                raise ValueError(f"{case.id} already passes at its checkpoint")
    return checkpoints


def run_case(game: GbaEmulator, case: Case, model: Model,
             checkpoints: dict[str, Checkpoint], output: Path,
             knowledge: str = "", model_spec: str = "injected") -> dict:
    output.mkdir(parents=True, exist_ok=False)
    log = RunLog(output, model_spec,
                 {"kind": "benchmark_case", "case": case.id, "start": case.start,
                  "criterion": case.criterion, "action_limit": case.limit,
                  "hold_frames": HOLD, "release_frames": RELEASE},
                 knowledge)
    watched = log.watch(model)
    agent = Agent(watched, watched, knowledge)
    agent.set_instructions(case.task)
    session = GameplaySession(game, output, HOLD, RELEASE)
    session.load(checkpoints[case.start])
    grader = Grader(case, game)
    events: list[str] = []
    started = time.monotonic()
    reason = "budget"
    error = None
    turns = 0
    reached_at = None
    while True:
        success, new_events = grader.observe(case.criterion)
        events.extend(new_events)
        if case.criterion in ("finish", "finish_lombre") and not success:
            reached_at = None  # A reopened the conversation after it had ended.
        elif success and reached_at is None:
            reached_at = turns
        if session.action_count >= case.limit:
            reason = "budget"
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
    public_reason = ("completion_claimed" if reason in ("recognized", "false_completion") else
                     "benchmark_limit" if reason == "budget" else reason)
    log.record_run_stop(public_reason, session.current_name)
    reached = reached_at is not None
    result = {"case": case.id, "task": case.task, "reached": reached,
              "recognized": reason == "recognized", "missed": reached and reason != "recognized",
              "false_completion": reason == "false_completion", "success": reached,
              "reached_at": reached_at, "stop_reason": reason,
              "actions": session.action_count, "turns": turns,
              "events": events,
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
    parser.add_argument("--model", default="mystery_dungeon.models.glm:GlmModel")
    parser.add_argument("--model-file", type=Path, action="append",
                        help="hash a local model artifact; repeat for weights and vision projector")
    parser.add_argument("--knowledge", type=Path, help="text the agent starts out knowing")
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    args = parser.parse_args()
    selected = [case for case in CASES if args.case is None or case.id in args.case]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases").mkdir()
    knowledge = args.knowledge.read_text() if args.knowledge else ""
    files = (args.rom, args.core, *(CHECKPOINTS / f"{name}.ss0" for name in
                                  ("square", "face-lombre", "face-npc", "npc-dialogue", "shop-interact")),
             *(args.model_file or ()),
             *((args.knowledge,) if args.knowledge else ()))
    hashes = {str(path): sha256(path) for path in files}
    hashes["model"] = args.model
    (args.output / "config.json").write_text(json.dumps({
        "format_version": 5, "layout": "benchmark", "grader": "pmd-red-ram-v1",
        "cases": [asdict(c) for c in selected],
        "hashes": hashes, "knowledge": str(args.knowledge) if args.knowledge else None,
        "hold_frames": HOLD, "release_frames": RELEASE,
    }, indent=2) + "\n")
    with GbaEmulator(args.rom, args.core) as game:
        checkpoints = build_fixtures(game, args.output)
        results = []
        for case in selected:
            print(case.id, flush=True)
            result = run_case(game, case, load_model(args.model), checkpoints,
                              args.output / "cases" / case.id, knowledge, args.model)
            results.append(result)
            report(results, args.output)
            print(f"  {result['stop_reason']} after {result['actions']} actions", flush=True)


if __name__ == "__main__":
    main()
