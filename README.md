# Mystery Dungeon Agent

A vision-based agent for Pokémon Mystery Dungeon: Red Rescue Team. Give it a short-term goal in the browser; it observes the game and chooses actions.

## Setup

Requires Python 3.12+, `uv`, a Red Rescue Team ROM, an mGBA Libretro core, and a save state. Set their paths in `config.toml`. Local OneJev also needs model files and `llama-server`.

## Run

```sh
uv run --extra onejev -m mystery_dungeon play
```

Open <http://127.0.0.1:8765> and enter a goal. For hosted GLM-5.3-Flash, put `OPENROUTER_API_KEY` in `.env` and run:

```sh
uv run --env-file .env -m mystery_dungeon play --decision-model mystery_dungeon.decision_models.glm:GlmDecisionModel
```

## Checks

```sh
uv run --extra onejev pytest -q
uv run --env-file .env -m mystery_dungeon.benchmark
```

The benchmark requires an OpenRouter API key and saves each run in a timestamped directory under `runs/benchmarks/`; pass `--output` to choose another path. Live sessions default to `runs/live/`. The one-step scripts read inputs from `runs/experiments/`; pass an output path there when running `scripts/probe.py`. Older scores in the linked documents describe earlier code; run a fresh benchmark to measure the current version. See [game context](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), and [historical benchmark results](docs/benchmark-vision-context.md).
