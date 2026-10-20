# Mystery Dungeon Agent

A vision-based agent for Pokémon Mystery Dungeon: Red Rescue Team. Give it a task in the browser, and it chooses actions from screenshots of the game.

## Setup

Install Python 3.12+ and `uv`. Provide a Red Rescue Team ROM, an mGBA Libretro core, and a save state, then set their paths under `[game]` in `config.toml`; `checkpoints` is the folder of benchmark save states. The agent asks a vision model behind any OpenAI-compatible chat-completions endpoint, set under `[model]` in `config.toml`; the default is GLM-5.3-Flash through OpenRouter, so put `OPENROUTER_API_KEY` in `.env`.

## Run

```sh
uv run --env-file .env -m mystery_dungeon.live
```

Open <http://127.0.0.1:8765> and enter a task. Input from the browser runs after the current agent turn. Add `--knowledge mystery_dungeon/knowledge/controls.md` to provide the game controls.

## How it works

On each turn, the model sees the task, the current screenshot and the last few turns with their screenshots, reasoning and actions. It reasons, then chooses a button press, a wait, or DONE when the task is complete. Live sessions save screenshots, game states, and event logs under `runs/live/`.

## Test

```sh
uv run python -m pytest -q
```

## Benchmark

```sh
uv run --env-file .env -m mystery_dungeon.benchmark
```

It records the model settings, memory length, an instructions hash and the hashes of the ROM, core, checkpoints and knowledge in `config.json`. Results are saved under `runs/benchmarks/`. For background, see [game context](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), [historical benchmark results](docs/benchmark-vision-context.md), and [reasoning window and single model](docs/reasoning-window-and-single-model.md).
