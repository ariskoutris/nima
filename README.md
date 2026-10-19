# Mystery Dungeon Agent

A vision-based agent for Pokémon Mystery Dungeon: Red Rescue Team. Give it a task in the browser, and it chooses actions from screenshots of the game.

## Setup

Install Python 3.12+ and `uv`. Provide a Red Rescue Team ROM, an mGBA Libretro core, and a save state, then set their paths under `[game]` in `config.toml`; `checkpoints` is the folder of benchmark save states. Local models need `llama-server` 0.7.0 or later; OneJev also needs its GGUF model files under `data/engines/models/`.

## Models

Each `[models.NAME]` table in `config.toml` names an adapter (`spec`, `MODULE:CLASS`) and a `label` for the browser's model picker, plus how to reach the model:

- `serve` and `ready`: a command the app runs on a free `{port}` (`{python}` is this project's interpreter) until the `ready` path answers, then stops when you switch away or quit. Relative paths are relative to `config.toml`; list local weights under `files` so benchmarks record their hashes.
- `url`: a server you run yourself, for example on another machine.
- neither: the adapter's own endpoint, such as OpenRouter for GLM.

The first table is the default. Run any configured server in the foreground with `uv run --extra onejev -m mystery_dungeon serve NAME --port 8080`.

## Run

```sh
uv run --extra onejev -m mystery_dungeon play
```

Open <http://127.0.0.1:8765> and enter a task. Intervening in the browser cancels the current agent turn.

Pass `--model NAME` to start with another configured model, or switch in the browser. Hosted GLM-5.3-Flash needs `OPENROUTER_API_KEY` in `.env`:

```sh
uv run --extra onejev --env-file .env -m mystery_dungeon play --model glm
```

`d1` (Liquid AI's d1-3B) downloads about 1.7 GB on its first start. Add `--knowledge mystery_dungeon/knowledge/controls.md` to provide the game controls.

## How it works

On each turn, the agent checks whether the task is complete, then chooses a button press or waits. It uses the current screenshot and recent actions to decide. Live sessions save screenshots, game states, and event logs under `runs/live/`.

## Test

```sh
uv run --extra onejev python -m pytest -q
```

## Benchmark

Name a configured model; `glm` needs `OPENROUTER_API_KEY` in `.env`:

```sh
uv run --extra onejev --env-file .env -m mystery_dungeon.benchmark --model glm
```

It serves local models itself and records the model's config table and the hashes of its `files` in `config.json`. Results are saved under `runs/benchmarks/`. For background, see [game context](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), and [historical benchmark results](docs/benchmark-vision-context.md).
