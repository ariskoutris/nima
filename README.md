# Mystery Dungeon Agent

A vision-based agent for Pokémon Mystery Dungeon: Red Rescue Team. Give it a task in the browser, and it chooses actions from screenshots of the game.

## Setup

Install Python 3.12+ and `uv`. Provide a Red Rescue Team ROM, an mGBA Libretro core, and a save state, then set their paths in `config.toml`. The default local OneJev model also needs GGUF model files and `llama-server`.

## Run

```sh
uv run --extra onejev -m mystery_dungeon play
```

Open <http://127.0.0.1:8765> and enter a task. Intervening in the browser cancels the current agent turn.

To use hosted GLM-5.3-Flash, put `OPENROUTER_API_KEY` in `.env` and run:

```sh
uv run --env-file .env -m mystery_dungeon play --model mystery_dungeon.models.glm:GlmModel
```

Add `--knowledge mystery_dungeon/knowledge/controls.md` to either command to provide the game controls.

## How it works

On each turn, the agent checks whether the task is complete, then chooses a button press or waits. It uses the current screenshot and recent actions to decide. Live sessions save screenshots, game states, and event logs under `runs/live/`.

## Test

```sh
uv run --extra onejev python -m pytest -q
```

## Benchmark

With `OPENROUTER_API_KEY` in `.env`, run:

```sh
uv run --env-file .env -m mystery_dungeon.benchmark
```

Results are saved under `runs/benchmarks/`. For background, see [game context](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), and [historical benchmark results](docs/benchmark-vision-context.md).
