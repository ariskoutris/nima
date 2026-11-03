"""Compare agent variants on the Pokémon Square game benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mystery_dungeon.benchmark import CASES  # noqa: E402 (needs ROOT on sys.path)
from mystery_dungeon.model import read_settings  # noqa: E402

GAME_FACTS = ROOT / "mystery_dungeon/knowledge/game-facts.md"
VARIANTS = {"baseline": None, "facts": GAME_FACTS}  # variant → knowledge file


def summarize(output: Path, model: str, variants: list[str], cases: list[str]) -> list[dict]:
    """Write summary.json and report.md for variants with results, and return their summaries."""
    summaries = []
    for variant in variants:
        result_file = output / variant / "results.json"
        if not result_file.is_file():
            continue
        rows = json.loads(result_file.read_text())
        latencies = []
        for row in rows:
            turns = output / variant / "cases" / row["case"] / "turns.jsonl"
            for line in turns.read_text().splitlines():
                turn = json.loads(line)
                if turn["event"] in ("action", "stop") and turn["source"] == "agent":
                    latencies.append(turn["model_ms"])
        summaries.append({
            "variant": variant, "cases": rows,
            "reached": sum(row["reached"] for row in rows),
            "claimed_reached": sum(row["claimed_reached"] for row in rows),
            "missed": sum(row["missed"] for row in rows),
            "claimed_unreached": sum(row["claimed_unreached"] for row in rows),
            "model_errors": sum(row["ended"] == "model_error" for row in rows),
            "median_turn_ms": round(statistics.median(latencies)) if latencies else None,
        })
    (output / "summary.json").write_text(json.dumps({
        "model": model, "selected_cases": cases, "variants": summaries,
    }, indent=2) + "\n")
    lines = ["# Probe comparison", "", f"Model: {model}", "",
             "| Variant | Cases | Reached | Claimed reached | Missed | Claimed unreached "
             "| Model errors | Median turn ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['variant']} | {len(item['cases'])}/{len(cases)} | "
                     f"{item['reached']} | {item['claimed_reached']} | {item['missed']} | "
                     f"{item['claimed_unreached']} | {item['model_errors']} | "
                     f"{item['median_turn_ms'] if item['median_turn_ms'] is not None else '—'} |")
    lines += ["", "## Case outcomes", "",
              "| Variant | Case | Reached | Claimed reached | Ended | Outcome | Actions "
              "| Seconds |",
              "|---|---|:---:|:---:|---|---|---:|---:|"]
    for item in summaries:
        for row in item["cases"]:
            lines.append(f"| {item['variant']} | {row['case']} | {row['reached']} | "
                         f"{row['claimed_reached']} | {row['ended']} | "
                         f"{row['outcome'] or '—'} | {row['actions']} | {row['seconds']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--output", type=Path,
                        help="new run directory (default: timestamped under runs/experiments)")
    args = parser.parse_args(argv)
    model = read_settings().name
    cases = [case.id for case in CASES if args.case is None or case.id in args.case]
    variants = list(dict.fromkeys(args.variant or VARIANTS))
    output = (args.output or ROOT / "runs/experiments" /
              f"probe-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({
        "model": model, "cases": cases, "variants": variants,
    }, indent=2) + "\n")
    try:
        for variant in variants:
            knowledge = VARIANTS[variant]
            command = [sys.executable, "-m", "mystery_dungeon.benchmark",
                       "--output", str(output / variant)]
            if knowledge:
                command += ["--knowledge", str(knowledge)]
            for case in cases:
                command += ["--case", case]
            print(variant, flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            summaries = summarize(output, model, variants, cases)
            if summaries[-1]["model_errors"]:
                raise SystemExit(f"{variant} had model errors; see {output / 'report.md'}")
    except subprocess.CalledProcessError as exc:
        summarize(output, model, variants, cases)
        raise SystemExit(exc.returncode) from exc
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
