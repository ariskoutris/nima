"""Replay fixed goals against OneJev with a grader hidden from the controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from emulator import Frame, GbaEmulator
from mystery_dungeon.agent import Agent, DecisionModel, load_decision_model
from mystery_dungeon.session import GameplaySession


ROM = Path("data/game/roms/red-rescue-team.gba")
CORE = Path("data/engines/cores/mgba_libretro.dylib")
STATE = Path("data/game/checkpoints/post-cutscene.ss0")
MODEL_FILES = (Path("data/engines/models/onejev-0.8b-q4_k_m.gguf"),
               Path("data/engines/models/onejev-0.8b-mmproj-q8_0.gguf"))
MAILBOX_ROUTE = ("RIGHT", "DOWN", "DOWN", "LEFT", "LEFT", "UP", "A")
HOLD, RELEASE, SECONDS = 8, 4, 180


@dataclass(frozen=True)
class Case:
    id: str
    start: str
    goal: str
    criterion: str
    limit: int


CASES = (
    Case("room_south", "inside", "Move one tile south", "south", 8),
    Case("room_exit_directional", "inside", "Walk down through the visible exit", "outside", 30),
    Case("room_exit_goal", "inside", "Get out of the base", "outside", 30),
    Case("outside_talk", "outside", "Talk to the nearby green Pokémon", "npc", 20),
    Case("npc_ready_talk", "npc_ready", "Talk to the Pokémon directly in front of you", "npc", 8),
    Case("outside_mailbox", "outside", "Check the mailbox", "mailbox", 40),
    Case("mailbox_front_check", "mailbox_front", "Check the mailbox in front of you", "mailbox", 8),
    Case("mailbox_front_button", "mailbox_front", "Press A to check the mailbox", "mailbox", 8),
    Case("room_then_talk", "inside", "Leave the base and talk to the nearby Pokémon", "npc", 60),
    Case("outside_clearing", "outside", "Find the wooded clearing beyond the base", "clearing", 70),
    Case("bridge_to_clearing", "bridge", "Cross the bridge north and reach the wooded clearing", "clearing", 25),
    Case("bridge_up_literal", "bridge", "Move up until you reach the wooded clearing", "clearing", 25),
    Case("mailbox_to_cave", "mailbox", "Follow the rescue request to the cave", "cave", 120),
    Case("outside_mailbox_to_cave", "outside", "Check the mailbox, then reach the cave", "mailbox_cave", 140),
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


def patch_distance(a: Frame, b: Frame, rect: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = rect
    total = count = 0
    for y in range(y0, y1, 2):
        for x in range(x0, x1, 2):
            offset = (y * a.width + x) * 4
            total += sum(abs(a.rgba[offset + i] - b.rgba[offset + i]) for i in range(3))
            count += 3
    return total / count


def colors(frame: Frame) -> tuple[int, int]:
    """Count forest and water palette pixels; no game pixels reach the agent."""
    forest = water = 0
    for offset in range(0, len(frame.rgba), 16):
        red, green, blue = frame.rgba[offset:offset + 3]
        forest += red < 90 and 40 <= green <= 200 and 45 <= blue <= 180
        water += red < 35 and green > 130 and blue > 210
    return forest, water


class Grader:
    def __init__(self, references: dict[str, Frame], initial_ram: bytes):
        self.references = references
        self.initial_y = initial_ram[2300]
        self.mailbox_seen = False

    def observe(self, frame: Frame, ram: bytes, criterion: str) -> tuple[bool, list[str]]:
        forest, water = colors(frame)
        mailbox = patch_distance(frame, self.references["mailbox"], (15, 126, 230, 153)) < 5
        npc = patch_distance(frame, self.references["npc"], (0, 60, 54, 120)) < 12
        cave = patch_distance(frame, self.references["cave"], (0, 0, 65, 50)) < 20
        milestones = []
        if mailbox and not self.mailbox_seen:
            self.mailbox_seen = True
            milestones.append("mailbox_checked")
        south = 8 <= (ram[2300] - self.initial_y) % 256 <= 80 and water < 200
        success = {
            "south": south,
            "outside": water > 500,
            "npc": npc,
            "mailbox": mailbox,
            "clearing": forest >= 4500 and water < 200,
            "cave": cave,
            "mailbox_cave": self.mailbox_seen and cave,
        }[criterion]
        return success, milestones


def build_fixtures(game: GbaEmulator, source: Seed, output: Path) -> tuple[dict[str, Seed], dict[str, Frame]]:
    seeds = {"outside": source}
    references = {}
    restore(game, source)
    references["outside"] = game.screenshot()

    restore(game, source)
    game.step(32, ("UP",))
    game.step(60)
    seeds["inside"] = Seed(game.save_state(), game.save_ram())
    references["inside"] = game.screenshot()

    restore(game, source)
    press(game, "DOWN")
    seeds["npc_ready"] = Seed(game.save_state(), game.save_ram())
    references["npc_ready"] = game.screenshot()
    press(game, "A")
    references["npc"] = game.screenshot()

    restore(game, source)
    press(game, "LEFT")
    press(game, "A")
    references["mailbox_side"] = game.screenshot()

    restore(game, source)
    for button in MAILBOX_ROUTE[:-1]:
        press(game, button)
    seeds["mailbox_front"] = Seed(game.save_state(), game.save_ram())
    references["mailbox_front"] = game.screenshot()
    press(game, "A")
    game.step(30)
    seeds["mailbox"] = Seed(game.save_state(), game.save_ram())
    references["mailbox"] = game.screenshot()

    for _ in range(30):
        game.step(HOLD, ("A",))
        game.step(40)
    references["cave"] = game.screenshot()

    restore(game, source)
    for _ in range(7):
        press(game, "RIGHT")
    seeds["bridge"] = Seed(game.save_state(), game.save_ram())
    references["bridge"] = game.screenshot()
    for _ in range(10):
        press(game, "UP")
    references["clearing"] = game.screenshot()

    refs = output / "references"
    refs.mkdir()
    for name, frame in references.items():
        frame.save_png(refs / f"{name}.png")
    for name, seed in seeds.items():
        (refs / f"{name}.state").write_bytes(seed.state)
        (refs / f"{name}.sav").write_bytes(seed.save_ram)
    validate_fixtures(game, source, seeds, references)
    return seeds, references


def validate_fixtures(game: GbaEmulator, source: Seed, seeds: dict[str, Seed], refs: dict[str, Frame]) -> None:
    """Fail before running a model if a reference or negative control is ambiguous."""
    assert colors(refs["outside"])[1] > 500
    assert colors(refs["inside"])[1] < 200
    assert colors(refs["clearing"])[0] >= 4500 and colors(refs["clearing"])[1] < 200
    assert patch_distance(refs["npc"], refs["outside"], (0, 60, 54, 120)) >= 12
    assert patch_distance(refs["npc"], refs["mailbox_side"], (0, 60, 54, 120)) >= 12
    assert patch_distance(refs["mailbox"], refs["mailbox_side"], (15, 126, 230, 153)) >= 5
    assert patch_distance(refs["cave"], refs["outside"], (0, 0, 65, 50)) >= 20
    assert colors(refs["bridge"])[0] < 4500
    for case in CASES:
        restore(game, seeds[case.start])
        initial = game.system_ram()
        grader = Grader(refs, initial)
        if case.start == "mailbox":
            grader.mailbox_seen = True
        assert not grader.observe(game.screenshot(), initial, case.criterion)[0], case.id
    restore(game, seeds["inside"])
    start = game.system_ram()
    press(game, "DOWN")
    assert Grader(refs, start).observe(game.screenshot(), game.system_ram(), "south")[0]
    restore(game, source)
    assert not Grader(refs, game.system_ram()).observe(game.screenshot(), game.system_ram(), "mailbox")[0]


RECOGNITION_WINDOW = 3


def run_case(game: GbaEmulator, case: Case, model: DecisionModel,
             seeds: dict[str, Seed], references: dict[str, Frame], output: Path,
             history_frames: int = 1, completion_votes: int = 1,
             context: str = "", model_spec: str = "injected") -> dict:
    output.mkdir(parents=True, exist_ok=False)
    restore(game, seeds[case.start])
    grader = Grader(references, game.system_ram())
    if case.start == "mailbox":
        grader.mailbox_seen = True
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
        frame = session.current
        success, new_events = grader.observe(frame, game.system_ram(), case.criterion)
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
    parser.add_argument("rom", type=Path, nargs="?", default=ROM)
    parser.add_argument("--core", type=Path, default=CORE)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-model", default="mystery_dungeon.decision_models.onejev:OneJevDecisionModel")
    parser.add_argument("--model-file", type=Path, action="append",
                        help="model artifact to hash; repeat for weights and vision projector")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--history-frames", type=int, default=1)
    parser.add_argument("--completion-votes", type=int, default=1)
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    args = parser.parse_args()
    if args.history_frames < 0 or args.completion_votes < 1:
        parser.error("history frames must be non-negative and completion votes positive")
    selected = [case for case in CASES if args.case is None or case.id in args.case]
    args.output.mkdir(parents=True, exist_ok=False)
    cases_dir = args.output / "cases"
    cases_dir.mkdir()
    source = read_ss0(args.state)
    context = args.context.read_text() if args.context else ""
    model_files = (MODEL_FILES if args.model_file is None and
                   args.decision_model == "mystery_dungeon.decision_models.onejev:OneJevDecisionModel"
                   else args.model_file or ())
    hashes = {str(path): sha256(path) for path in (args.rom, args.core, args.state, *model_files,
                                                 *((args.context,) if args.context else ()))}
    hashes["decision_model"] = args.decision_model
    (args.output / "config.json").write_text(json.dumps({
        "format_version": 2, "layout": "benchmark",
        "cases": [asdict(c) for c in selected], "hashes": hashes,
        "context": str(args.context) if args.context else None,
        "history_frames": args.history_frames, "completion_votes": args.completion_votes,
        "recognition_window": RECOGNITION_WINDOW,
        "hold_frames": HOLD, "release_frames": RELEASE}, indent=2) + "\n")
    with GbaEmulator(args.rom, args.core) as game:
        seeds, refs = build_fixtures(game, source, args.output)
        results = []
        for case in selected:
            print(case.id, flush=True)
            result = run_case(game, case, load_decision_model(args.decision_model), seeds, refs,
                              cases_dir / case.id, args.history_frames, args.completion_votes,
                              context, args.decision_model)
            results.append(result)
            report(results, args.output)
            print(f"  {result['stop_reason']} after {result['steps']} actions", flush=True)


if __name__ == "__main__":
    main()
