# Mystery Dungeon agent

An agent that plays Pokémon Mystery Dungeon: Red Rescue Team (GBA) toward a goal. It observes game frames, chooses actions with a vision decision model, and uses the results to choose its next action. The operator currently supplies each short-term goal; goal selection and planning can become agent capabilities later.

## Setup and run

Requires `uv`, `llama-server`, an mGBA Libretro core, a ROM, and a save state under `data/`. Their default paths are in `config.toml`. From the repo root, run these in separate terminals:

```sh
uv run --extra onejev -m mystery_dungeon serve
```

```sh
uv run --extra onejev -m mystery_dungeon play
```

Open <http://127.0.0.1:8765> and enter a goal. A goal change during inference takes effect after the current action. Append `--context mystery_dungeon/prompts/controls-v1.md` to the play command for the optional control guide. Other options include `--history-frames N` (default 1), `--completion-votes N` (default 1), and `--decision-model module:Class` for another no-argument model class.

## Benchmark and tests

```sh
uv run --extra onejev python -m mystery_dungeon.benchmark --output runs/benchmark/example
uv run --extra onejev pytest -q
```

The benchmark grades goal reach independently of the decision model. Results are in `report.md` and `results.json`; each case has its own journal and screenshots under `cases/<case-id>/`. Use `--case <case-id>` for a single case. `python -m mystery_dungeon.play --help` lists replay options.

## Code and data

- `emulator.py` wraps the Libretro core.
- `mystery_dungeon/agent.py` owns each gameplay turn: choose an action or stop, apply it, track the result, and record experience.
- `mystery_dungeon/session.py` provides emulator I/O: apply inputs and capture resulting frames.
- `mystery_dungeon/decision_models/` contains system-one adapters that score actions and visible goal completion. A model implements `score(goal, guidance, current_frame, recent_steps, button_options) -> DecisionScores`.
- `mystery_dungeon/live.py`, `play.py`, and `benchmark.py` are the runners; `scripts/` holds one-off experiments.

Generated data belongs in ignored `runs/`. A run contains `config.json`, `decisions.jsonl`, numbered `frames/`, `states/`, and snapshots of any loaded prompt or guide. OneJev loads `mystery_dungeon/prompts/onejev-v1.json`; set `QEV_PROMPT_VERSION=vN` to select another immutable version. Run configs record prompt, guide, and button-option hashes. Historical runs retain their earlier layout.

Further context: [game planning reference](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), and [historical vision benchmark](docs/benchmark-vision-context.md).
