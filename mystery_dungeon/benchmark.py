"""Run short Pokémon Square goals with hidden game-state grading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from emulator import GbaEmulator
from mystery_dungeon.agent import Agent, DecisionModel, load_decision_model
from mystery_dungeon.session import GameplaySession


ROM = Path("data/game/roms/red-rescue-team.gba")
CORE = Path("data/engines/cores/mgba_libretro.dylib")
CHECKPOINTS = Path("data/game/checkpoints")
ROM_SHA1 = "9f4cfc5b5f4859d17169a485462e977c7aac2b89"
HOLD, RELEASE, SECONDS = 8, 4, 180
RECOGNITION_WINDOW = 3


@dataclass(frozen=True)
class Case:
    id: str
    start: str
    goal: str
    criterion: str
    limit: int


CASES = (
    Case("square_down", "square", "Move one tile down", "down", 5),
    Case("square_north_path", "square",
         "Walk north until the central plaza is out of view", "north", 14),
    Case("square_approach_granbull", "square",
         "Stand just above the purple Pokémon on the left side of the square", "approach", 12),
    Case("square_talk_granbull", "square",
         "Walk to Granbull in the square and start a conversation", "granbull", 16),
    Case("face_npc_talk", "face-npc", "Start talking to the Pokémon in front of you", "granbull", 5),
    Case("dialogue_finish", "npc-dialogue", "Finish the conversation with Granbull", "finish", 18),
    Case("face_npc_finish", "face-npc", "Talk to Granbull until the conversation ends", "finish", 22),
    Case("shop_talk", "shop-interact", "Talk to the shopkeeper in front of you", "wigglytuff", 5),
    Case("shop_friend_areas", "shop-interact", "Open the Friend Areas list at the shop", "shop_menu", 22),
)


@dataclass(frozen=True)
class Seed:
    state: bytes
    save_ram: bytes


def read_ss0(path: Path) -> Seed:
    """Extract the mGBA core state and battery RAM from its PNG save state."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("checkpoint is not a PNG/mGBA save state")
    state = save_ram = None
    offset = 8
    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError("truncated mGBA save state")
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + payload) != crc:
            raise ValueError(f"bad {kind!r} checksum")
        if kind == b"gbAs":
            state = zlib.decompress(payload)
        elif kind == b"gbAx" and payload[:4] == b"\x02\0\0\0":
            save_ram = zlib.decompress(payload[8:])
        offset = end
    if state is None or save_ram is None:
        raise ValueError("mGBA state or save RAM is missing")
    return Seed(state, save_ram)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore(game: GbaEmulator, seed: Seed) -> None:
    game.load_save_ram(seed.save_ram)
    game.load_state(seed.state)
    game.step()


def press(game: GbaEmulator, button: str) -> None:
    game.step(HOLD, () if button == "WAIT" else (button,))
    game.step(RELEASE)


class Grader:
    """Read fields documented by https://github.com/pret/pmd-red for ROM_SHA1."""

    def __init__(self, case: Case, game: GbaEmulator):
        self.game = game
        position = self.player_position()
        if position is None:
            raise ValueError(f"player position unavailable in {case.start} checkpoint")
        self.initial_x, self.initial_y = position
        self.saw_granbull = self.dialogue() == (1, 235)

    def value(self, address: int, fmt: str) -> int:
        return struct.unpack(fmt, self.game.read_memory(address, struct.calcsize(fmt)))[0]

    def player_position(self) -> tuple[int, int] | None:
        # gGroundLives -> GroundLive[0].unk144 (PixelPos, s24.8).
        lives = self.value(0x03001B84, "<I")
        if not 0x02000000 <= lives < 0x02040000:
            return None
        x, y = struct.unpack("<ii", self.game.read_memory(lives + 0x144, 8))
        return x // 256, y // 256

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

    def observe(self, criterion: str) -> tuple[bool, list[str]]:
        position = self.player_position()
        if position is None:
            return False, []
        x, y = position
        dialogue, speaker = self.dialogue()
        granbull = dialogue == 1 and speaker == 235
        if granbull:
            self.saw_granbull = True
        npc = self.granbull_position() if criterion == "approach" else None
        success = {
            "down": x == self.initial_x and y >= self.initial_y + 16,
            "north": y <= self.initial_y - 128,
            "approach": (npc is not None and abs(x - npc[0]) <= 12 and
                         8 <= npc[1] - y <= 24 and dialogue == 0),
            "granbull": granbull,
            "finish": self.saw_granbull and dialogue == 0,
            "wigglytuff": dialogue == 4 and speaker == 40,
            "shop_menu": self.shop_state() in (9, 10) if criterion == "shop_menu" else False,
        }[criterion]
        return success, []


def build_fixtures(game: GbaEmulator, output: Path) -> dict[str, Seed]:
    if hashlib.sha1(game.rom.read_bytes()).hexdigest() != ROM_SHA1:
        raise ValueError("town RAM offsets require the matching Red Rescue Team ROM")
    seeds = {name: read_ss0(CHECKPOINTS / f"{name}.ss0")
             for name in ("square", "face-npc", "npc-dialogue", "shop-interact")}
    folder = output / "references"
    folder.mkdir()
    for name, seed in seeds.items():
        (folder / f"{name}.state").write_bytes(seed.state)
        (folder / f"{name}.sav").write_bytes(seed.save_ram)
        restore(game, seed)
        game.screenshot().save_png(folder / f"{name}.png")
        for case in (case for case in CASES if case.start == name):
            grader = Grader(case, game)
            if grader.observe(case.criterion)[0]:
                raise ValueError(f"{case.id} already passes at its checkpoint")
    return seeds


def run_case(game: GbaEmulator, case: Case, model: DecisionModel,
             seeds: dict[str, Seed], output: Path,
             history_frames: int = 1, completion_votes: int = 1,
             context: str = "", model_spec: str = "injected") -> dict:
    output.mkdir(parents=True, exist_ok=False)
    restore(game, seeds[case.start])
    grader = Grader(case, game)
    agent = Agent(model, output, model_spec, history_frames, completion_votes, context,
                  {"kind": "benchmark_case", "case": case.id, "start": case.start,
                   "criterion": case.criterion, "action_limit": case.limit,
                   "hold_frames": HOLD, "release_frames": RELEASE},
                  detect_cycles=True, transition_wait=True)
    agent.set_goal(case.goal)
    session = GameplaySession(game, output, HOLD, RELEASE)
    session.capture()
    events: list[str] = []
    started = time.monotonic()
    reason = "budget"
    error = None
    actions = decisions = 0
    reached_at = None
    while True:
        success, new_events = grader.observe(case.criterion)
        events.extend(new_events)
        if success and reached_at is None:
            reached_at = decisions
        if time.monotonic() - started >= SECONDS:
            reason = "time_limit"
            break
        if reached_at is None and actions >= case.limit:
            reason = "budget"
            break
        if reached_at is not None and decisions - reached_at >= RECOGNITION_WINDOW:
            reason = "recognition_window"
            break
        turn = agent.step(session, deadline=started + SECONDS)
        if turn.status == "model_error":
            reason, error = "model_error", turn.error
            break
        decisions += 1
        if turn.status == "time_limit":
            reason = "time_limit"
            break
        if turn.status == "completed":
            reason = "recognized" if reached_at is not None else "false_completion"
            break
        if turn.status in ("stalled", "loop"):
            reason = turn.status
            break
        if turn.status == "acted":
            actions += 1
    final = session.current
    final.save_png(output / "frames/final.png")
    (output / "states").mkdir(exist_ok=True)
    (output / "states/final.state").write_bytes(game.save_state())
    (output / "states/final.sav").write_bytes(game.save_ram())
    public_reason = ("completion_claimed" if reason in ("recognized", "false_completion") else
                     "benchmark_limit" if reason in ("budget", "time_limit", "recognition_window") else reason)
    agent.record({"event": "run_stop", "reason": public_reason,
                  "screenshot": session.current_name})
    reached = reached_at is not None
    result = {"case": case.id, "goal": case.goal, "reached": reached,
              "recognized": reason == "recognized", "missed": reached and reason != "recognized",
              "false_completion": reason == "false_completion", "success": reached,
              "reached_at": reached_at, "stop_reason": reason,
              "steps": actions, "decisions": decisions,
              "low_visual_change_actions": agent.low_visual_change_actions,
              "transition_waits": agent.transition_waits, "events": events,
              "seconds": round(time.monotonic() - started, 2), "error": error}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def report(results: list[dict], output: Path) -> None:
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Game goal benchmark", "",
             "| Case | Reached | Recognized | Missed | False completion | Actions | Seconds |",
             "|---|:---:|:---:|:---:|:---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['case']} | {row['reached']} | {row['recognized']} | "
                     f"{row['missed']} | {row['false_completion']} | {row['steps']} | {row['seconds']} |")
    lines += ["", "The hidden grader measures reach independently from model completion.",
              "Joint action/completion requests and frame history change prompts and latency; "
              "these runs are not directly comparable with earlier benchmark reports.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", type=Path, default=ROM)
    parser.add_argument("--core", type=Path, default=CORE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-model", default="mystery_dungeon.decision_models.glm:GlmDecisionModel")
    parser.add_argument("--expected-model", help="require this model name from a local OneJev server")
    parser.add_argument("--model-file", type=Path, action="append",
                        help="hash a local model artifact; repeat for weights and vision projector")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--history-frames", type=int, default=1)
    parser.add_argument("--completion-votes", type=int, default=1)
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    args = parser.parse_args()
    if args.history_frames < 0 or args.completion_votes < 1:
        parser.error("history frames must be non-negative and completion votes positive")
    model_server = None
    if args.expected_model:
        if args.decision_model != "mystery_dungeon.decision_models.onejev:OneJevDecisionModel":
            parser.error("--expected-model requires the OneJev decision model")
        base_url = (os.getenv("QEV_BASE_URL") or os.getenv("TYPESAFE_BASE_URL") or
                    "http://localhost:8000")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=5) as response:
                model_server = json.load(response)
        except (OSError, ValueError) as exc:
            parser.error(f"could not check OneJev server: {exc}")
        reported = model_server.get("model") if isinstance(model_server, dict) else None
        if reported != args.expected_model:
            parser.error(f"OneJev server reports {reported!r}, expected {args.expected_model!r}")
    selected = [case for case in CASES if args.case is None or case.id in args.case]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases").mkdir()
    context = args.context.read_text() if args.context else ""
    files = (args.rom, args.core, *(CHECKPOINTS / f"{name}.ss0" for name in
                                  ("square", "face-npc", "npc-dialogue", "shop-interact")),
             *(args.model_file or ()),
             *((args.context,) if args.context else ()))
    hashes = {str(path): sha256(path) for path in files}
    hashes["decision_model"] = args.decision_model
    (args.output / "config.json").write_text(json.dumps({
        "format_version": 3, "layout": "benchmark", "grader": "pmd-red-ram-v1",
        "cases": [asdict(c) for c in selected],
        "hashes": hashes, "context": str(args.context) if args.context else None,
        "history_frames": args.history_frames, "completion_votes": args.completion_votes,
        "model_server": model_server,
        "hold_frames": HOLD, "release_frames": RELEASE, "time_limit_seconds": SECONDS,
        "recognition_window": RECOGNITION_WINDOW,
    }, indent=2) + "\n")
    with GbaEmulator(args.rom, args.core) as game:
        seeds = build_fixtures(game, args.output)
        results = []
        for case in selected:
            print(case.id, flush=True)
            result = run_case(game, case, load_decision_model(args.decision_model), seeds,
                              args.output / "cases" / case.id,
                              args.history_frames, args.completion_votes, context,
                              args.decision_model)
            results.append(result)
            report(results, args.output)
            print(f"  {result['stop_reason']} after {result['steps']} actions", flush=True)


if __name__ == "__main__":
    main()
