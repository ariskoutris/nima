# Mystery Dungeon agent

A vision driven agent for Pokémon Mystery Dungeon: Red Rescue Team (GBA). The mGBA Libretro core supplies frames and executes inputs. A system-one decision model scores buttons and visible goal completion; an operator supplies the current short-term goal.

## Setup and run

Requires `uv`, `llama-server`, an mGBA Libretro core, a ROM, and a save state under `data/`. Start the OneJev server, then the live controller in separate terminals:

```sh
uv run --extra onejev qev serve \
  --gguf data/engines/models/onejev-0.8b-q4_k_m.gguf \
  --mmproj data/engines/models/onejev-0.8b-mmproj-q8_0.gguf
```

```sh
uv run --extra onejev python -m mystery_dungeon.live \
  --core data/engines/cores/mgba_libretro.dylib \
  --state data/game/checkpoints/post-tiny-woods.state \
  data/game/roms/red-rescue-team.gba
```

Open <http://127.0.0.1:8765> and enter a goal. `--context mystery_dungeon/prompts/controls-v1.md` adds the optional control guide. Use `--history-frames N` (default 1) and `--completion-votes N` (default 1) to tune memory and completion confirmation. `--decision-model module:Class` selects another no-argument model class.

## Benchmark and tests

```sh
uv run --extra onejev python -m mystery_dungeon.benchmark --output runs/benchmark/example
uv run --extra onejev pytest -q
```

The benchmark grades goal reach independently of the decision model. Results are in `report.md` and `results.json`; each case has its own journal and screenshots under `cases/<case-id>/`. Use `--case <case-id>` for a single case. `python -m mystery_dungeon.play --help` lists replay options.

## Code and data

- `emulator.py` wraps the Libretro core.
- `mystery_dungeon/agent.py` owns goals, short visual memory, and the append-only decision journal.
- `mystery_dungeon/session.py` applies inputs and captures frames.
- `mystery_dungeon/decision_models/` contains system-one adapters. A model implements `score(goal, guidance, current_frame, recent_steps, button_options) -> DecisionScores`.
- `mystery_dungeon/live.py`, `play.py`, and `benchmark.py` are the runners; `scripts/` holds one-off experiments.

Generated data belongs in ignored `runs/`. A run contains `config.json`, `decisions.jsonl`, numbered `frames/`, `states/`, and snapshots of any loaded prompt or guide. OneJev loads `mystery_dungeon/prompts/onejev-v1.json`; set `QEV_PROMPT_VERSION=vN` to select another immutable version. Run configs record prompt, guide, and button-option hashes. Historical runs retain their earlier layout.

Further context: [game planning reference](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), and [historical vision benchmark](docs/benchmark-vision-context.md).
