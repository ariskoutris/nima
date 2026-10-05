"""One-shot prompt ablations on fixed, labeled game screenshots."""

import argparse
import base64
import io
import json
import time
from pathlib import Path

from PIL import Image
from qev import Choice, Client

ROOT = Path("runs/benchmark-20261004-4b-plain/references")
CONTROLS = Path("mystery_dungeon/prompts/controls-v1.md").read_text().strip()
QUESTION = "Which gamepad action should be taken next to make progress?"
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
ORDERED = {key: OPTIONS[key] for key in
           ("A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R", "WAIT")}
SMALL = {key: OPTIONS[key] for key in ("A", "B", "UP", "DOWN", "LEFT", "RIGHT", "WAIT")}
MANUAL = {
    "UP": "move one step up in the world, or move the menu cursor up",
    "DOWN": "move one step down in the world, or move the menu cursor down",
    "LEFT": "move one step left in the world, or move the menu cursor left",
    "RIGHT": "move one step right in the world, or move the menu cursor right",
    "A": "press the A button to interact with or talk to what the character faces; confirm a menu choice; advance visible text or dialogue; attack in a dungeon",
    "B": "press the B button to open the menu during play, cancel a command, or go back",
    "START": "press START to change facing in a dungeon; unused in the field",
    "SELECT": "press SELECT to show the dungeon map",
    "L": "press the left shoulder button",
    "R": "press the right shoulder button",
    "WAIT": "press nothing and let the game advance",
}
CASES = {
    "room_down": ("inside.png", "Move one tile down toward the visible opening", "DOWN"),
    "room_goal": ("inside.png", "Get out of the base", "DOWN"),
    "room_literal": ("inside.png", "Press DOWN once", "DOWN"),
    "npc_talk": ("npc_ready.png", "Talk to the Pokemon directly in front of you", "A"),
    "mailbox_check": ("mailbox_front.png", "Check the mailbox in front of you", "A"),
    "mailbox_literal": ("mailbox_front.png", "Press A once to check the mailbox", "A"),
    "bridge_right": ("outside.png", "Move one tile right toward the wooden bridge", "RIGHT"),
    "bridge_up": ("bridge.png", "Move one tile up", "UP"),
}
VARIANTS = (
    "current", "canonical", "ordered", "small", "debias3", "debias11", "image3x",
    "controls_field", "controls_inline", "current_guide", "current_3x", "current_guide_3x",
    "manual_options", "manual_ordered", "manual_small", "manual_question",
    "manual_small_2x", "manual_small_3x", "manual_ordered_3x",
)


def image_uri(path: Path, scale: int) -> str:
    if scale > 1:
        with Image.open(path) as picture:
            picture = picture.resize((picture.width * scale, picture.height * scale), Image.Resampling.NEAREST)
            buffer = io.BytesIO()
            picture.save(buffer, format="PNG")
            data = buffer.getvalue()
    else:
        data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def request(client: Client, case: str, variant: str, cases: dict) -> dict:
    filename, goal, expected = cases[case]
    if variant in ("current", "current_guide", "current_3x", "current_guide_3x",
                   "manual_options", "manual_ordered", "manual_small", "manual_question",
                   "manual_small_2x", "manual_small_3x", "manual_ordered_3x"):
        state = {"screen": "<image:1>", "task": {"instruction": goal}}
    else:
        state = {"task": goal, "screen": "<image:1>"}
    if variant == "controls_field":
        state["controls"] = CONTROLS
    if variant == "controls_inline":
        state["task"] = CONTROLS + "\n\nCurrent goal: " + goal
    if variant in ("current_guide", "current_guide_3x"):
        state["task"]["instruction"] = CONTROLS + "\n\nCurrent goal: " + goal
    options = (ORDERED if variant in ("ordered", "image3x", "controls_field", "controls_inline")
               else SMALL if variant == "small" else OPTIONS)
    if variant in ("manual_options", "manual_question"):
        options = MANUAL
    elif variant in ("manual_ordered", "manual_ordered_3x"):
        options = {key: MANUAL[key] for key in ORDERED}
    elif variant in ("manual_small", "manual_small_2x", "manual_small_3x"):
        options = {key: MANUAL[key] for key in SMALL}
    question = ("Given the screenshot and task, which gamepad button should be pressed now? "
                "The option letters are answer labels, not gamepad buttons."
                if variant == "manual_question" else QUESTION)
    started = time.perf_counter()
    scale = (2 if variant == "manual_small_2x" else
             3 if variant in ("image3x", "current_3x", "current_guide_3x",
                              "manual_small_3x", "manual_ordered_3x") else 1)
    answer = client.system_one(
        state=state,
        media=[{"type": "image", "data": image_uri(Path(filename), scale)}],
        questions={"next": Choice(question, options)},
        **({"debias": 3 if variant == "debias3" else 11} if variant in ("debias3", "debias11") else {}),
    )
    scores = dict(answer.answers["next"].probabilities)
    choice = max(scores, key=scores.get)
    return {"case": case, "variant": variant, "expected": expected, "choice": choice,
            "correct": choice == expected, "expected_score": scores[expected],
            "latency_s": round(time.perf_counter() - started, 2), "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="JSON manifest of held-out screenshots and labels")
    parser.add_argument("--case", action="append")
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    args = parser.parse_args()
    if args.manifest:
        rows = json.loads(args.manifest.read_text())
        cases = {key: (str(args.manifest.parent / row["image"]), row["goal"], row["expected"])
                 for key, row in rows.items()}
    else:
        cases = {key: (str(ROOT / image), goal, expected)
                 for key, (image, goal, expected) in CASES.items()}
    if args.case and set(args.case) - cases.keys():
        parser.error("unknown case: " + ", ".join(sorted(set(args.case) - cases.keys())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = Client()
    with args.output.open("w") as output:
        for variant in args.variant or VARIANTS:
            for case in args.case or cases:
                row = request(client, case, variant, cases)
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(variant, case, row["choice"], "OK" if row["correct"] else "FAIL",
                      row["latency_s"], flush=True)


if __name__ == "__main__":
    main()
