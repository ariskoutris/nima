# Mystery Dungeon Agent

A vision-based agent for Pokémon Mystery Dungeon: Red Rescue Team. Give it instructions in the browser; it observes the game, plans a task and chooses actions.

## Setup

Requires Python 3.12+, `uv`, a Red Rescue Team ROM, an mGBA Libretro core, and a save state. Set their paths in `config.toml`. Local OneJev also needs model files and `llama-server`.

## Run

```sh
uv run --extra onejev -m mystery_dungeon play
```

Open <http://127.0.0.1:8765> and enter instructions. Pausing, a manual action, a reset, loading a checkpoint, switching models or editing instructions discards the agent's turn in progress, so no button is pressed after you intervene. For hosted GLM-5.3-Flash, put `OPENROUTER_API_KEY` in `.env` and run:

```sh
uv run --env-file .env -m mystery_dungeon play --model mystery_dungeon.models.glm:GlmModel
```

Add `--knowledge mystery_dungeon/knowledge/controls.md` to start the agent with the game's control guide.

## How it works

Each turn the agent observes a screenshot, plans which task to pursue (today: the operator's instructions, plus a check of whether that task is complete), and acts by choosing one action from `mystery_dungeon/actions.py` (`Press(button)` or `Wait()`). A model fills two roles defined in `mystery_dungeon/models/`: `is_complete(task, frame, memory) -> bool` and `choose_action(task, knowledge, frame, memory) -> Action`. Memory holds the latest screenshot with the action taken after it. GLM and OneJev share the wording in `mystery_dungeon/prompts.py`.

Each run directory holds `config.json`, `events.jsonl` (one row per turn, discarded turn, manual action, task change, reset, model change or run stop, including each model call's duration and raw response), numbered `frames/` and `states/`.

## Checks

```sh
uv run --extra onejev pytest -q
uv run --env-file .env -m mystery_dungeon.benchmark
```

The benchmark requires an OpenRouter API key and saves each run in a timestamped directory under `runs/benchmarks/`; pass `--output` to choose another path, `--model` for another model and `--knowledge` for starting knowledge. Live sessions default to `runs/live/`. The one-step scripts read inputs from `runs/experiments/`; pass an output path there when running `scripts/probe.py`. Older scores in the linked documents describe earlier code; run a fresh benchmark to measure the current version. See [game context](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), and [historical benchmark results](docs/benchmark-vision-context.md).
