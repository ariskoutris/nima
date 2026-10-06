"""Compare production decision models on the Pokémon Square game benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mystery_dungeon.benchmark import CASES
from mystery_dungeon.live import ModelServer


CONTROL = ROOT / "mystery_dungeon/prompts/control.md"
MODELS = {
    "glm": ("mystery_dungeon.decision_models.glm:GlmDecisionModel", ()),
    "jev-0.8b": ("mystery_dungeon.decision_models.onejev:OneJevDecisionModel", (
        ROOT / "data/engines/models/onejev-0.8b-q4_k_m.gguf",
        ROOT / "data/engines/models/onejev-0.8b-mmproj-q8_0.gguf",
    )),
    "jev-4b": ("mystery_dungeon.decision_models.onejev:OneJevDecisionModel", (
        ROOT / "data/engines/models/OneJev-4B.Q4_K_M.gguf",
        ROOT / "data/engines/models/OneJev-4B.mmproj-Q8_0.gguf",
    )),
}
VARIANTS = {
    "baseline": (1, 1, None),
    "no_history": (0, 1, None),
    "controls": (1, 1, CONTROL),
    "image2x": (1, 2, None),
}


def summarize(output: Path, model: str, variants: list[str], cases: list[str]) -> list[dict]:
    summaries = []
    for variant in variants:
        result_file = output / variant / "results.json"
        if not result_file.is_file():
            continue
        rows = json.loads(result_file.read_text())
        latencies = []
        for row in rows:
            journal = output / variant / "cases" / row["case"] / "decisions.jsonl"
            for line in journal.read_text().splitlines():
                decision = json.loads(line)
                if decision["event"] in ("action", "completion"):
                    latencies.append(decision["latency_ms"])
        summaries.append({
            "variant": variant, "cases": rows,
            "reached": sum(row["reached"] for row in rows),
            "recognized": sum(row["recognized"] for row in rows),
            "missed": sum(row["missed"] for row in rows),
            "false_completion": sum(row["false_completion"] for row in rows),
            "model_errors": sum(row["stop_reason"] == "model_error" for row in rows),
            "median_decision_ms": round(statistics.median(latencies)) if latencies else None,
        })
    (output / "summary.json").write_text(json.dumps({
        "model": model, "selected_cases": cases, "variants": summaries,
    }, indent=2) + "\n")
    lines = ["# Probe comparison", "", f"Model: {model}", "",
             "| Configuration | Cases | Reached | Recognized | Missed | False completion | Model errors | Median decision ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['variant']} | {len(item['cases'])}/{len(cases)} | "
                     f"{item['reached']} | {item['recognized']} | {item['missed']} | "
                     f"{item['false_completion']} | {item['model_errors']} | "
                     f"{item['median_decision_ms'] if item['median_decision_ms'] is not None else '—'} |")
    lines += ["", "## Case outcomes", "",
              "| Configuration | Case | Reached | Recognized | Stop reason | Actions | Seconds |",
              "|---|---|:---:|:---:|---|---:|---:|"]
    for item in summaries:
        for row in item["cases"]:
            lines.append(f"| {item['variant']} | {row['case']} | {row['reached']} | "
                         f"{row['recognized']} | {row['stop_reason']} | {row['steps']} | "
                         f"{row['seconds']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--output", type=Path, help="new run directory (default: timestamped under runs/experiments)")
    args = parser.parse_args(argv)
    model_spec, model_files = MODELS[args.model]
    cases = [case.id for case in CASES if args.case is None or case.id in args.case]
    variants = list(dict.fromkeys(args.variant or VARIANTS))
    if args.model == "glm" and not os.getenv("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for GLM")
    for path in model_files:
        if not path.is_file():
            parser.error(f"local model file not found: {path}")
    output = (args.output or ROOT / "runs/experiments" /
              f"probe-{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({
        "model": args.model, "decision_model": model_spec,
        "model_files": [str(path) for path in model_files],
        "cases": cases, "variants": variants,
    }, indent=2) + "\n")
    server = ModelServer(str(model_files[0]), str(model_files[1]), output) if model_files else None
    environment = os.environ.copy()
    try:
        if server:
            environment["QEV_BASE_URL"] = server.start()
        for variant in variants:
            history, scale, context = VARIANTS[variant]
            command = [sys.executable, "-m", "mystery_dungeon.benchmark",
                       "--output", str(output / variant),
                       "--decision-model", model_spec,
                       "--history-frames", str(history),
                       "--image-scale", str(scale),
                       "--completion-votes", "1"]
            if context:
                command += ["--context", str(context)]
            else:
                command += ["--no-context"]
            for path in model_files:
                command += ["--model-file", str(path)]
            for case in cases:
                command += ["--case", case]
            print(f"{args.model}: {variant}", flush=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            summaries = summarize(output, args.model, variants, cases)
            if summaries[-1]["model_errors"]:
                raise SystemExit(f"{variant} had model errors; see {output / 'report.md'}")
    except subprocess.CalledProcessError as exc:
        summarize(output, args.model, variants, cases)
        raise SystemExit(exc.returncode) from exc
    finally:
        if server:
            server.close()
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
