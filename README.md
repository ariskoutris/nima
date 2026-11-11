# nima

An agent that plays Pokémon Mystery Dungeon: Red Rescue Team using a vision model. Give it a task such as talking to a character or buying an item, and watch it play in the browser. You can follow its reasoning and take control between turns.

## The agent loop

The agent is given a task and operates in a simple loop:

1. Observe the current game state, consisting of the frame and information derived from RAM (e.g. player position).
2. Think about what to do next to complete the task.
3. Choose an action: press a button, wait, or stop and mark the task as complete or failed.

To keep the model's context small, only the five most recent turns are kept in memory. To preserve information that is useful in the long term (e.g. completed parts of a task, failed attempts, useful dialogue), the agent maintains a notepad that it can write to each turn. It can then refer to this notepad in later turns.

[WIP] A higher-level orchestrator prioritizes and assigns tasks and reconciles the agent’s notepad entries into long-term memory. This higher level function is necessary to act autonomously for hours, to make decisions informed by past experience and to choose what to explore next.

## Setup

With Python 3.12+ and `uv` installed, run this from the project directory to install the dependencies:

```sh
uv sync
```

Then point the `[game]` section of `config.toml` to your own Red Rescue Team ROM, mGBA Libretro core and starting save state.

The `[model]` section selects the vision model and its OpenAI-compatible endpoint. It defaults to GLM-5.3-Flash through OpenRouter, whose API key goes in `.env` as `OPENROUTER_API_KEY`.

## Run

```sh
uv run --env-file .env -m nima.live
```

Open <http://127.0.0.1:8765> and enter a task. The browser shows the game, reasoning, actions and notes. You can pause the agent, press buttons yourself or load a save state. Sessions are saved under `runs/live/`.

## Benchmark

```sh
uv run --env-file .env -m nima.benchmark
```

The benchmark tests various tasks covering movement, conversations and shop menus. It grades progress from RAM independently of the model's completion claim. Use `--case ID` to run one case.

Results, frames, save states and model responses are saved under `runs/benchmarks/`. Each run records its settings and input hashes in `config.json`.

## Development

```sh
uv run python -m pytest -q
uv run ruff check
```

The browser UI tests require Node.js:

```sh
node --test tests/live_ui.test.cjs
```
