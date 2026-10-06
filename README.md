# Mystery Dungeon agent

An agent that plays Pokémon Mystery Dungeon: Red Rescue Team (GBA) toward a goal. It observes game frames, chooses actions with a vision decision model, and uses the results to choose its next action. The operator currently supplies each short-term goal; goal selection and planning can become agent capabilities later.

## Setup and run

Requires `uv`, an mGBA Libretro core, a ROM, and a save state under `data/`. Local OneJev play also requires `llama-server`. Default paths are in `config.toml`. From the repo root, start the live platform:

```sh
uv run --extra onejev -m mystery_dungeon play
```

Open <http://127.0.0.1:8765> and enter a goal. The **Live game** panel streams frames as the emulator advances and shows when it is waiting for a model response; the game pauses during that wait. The live platform starts the configured OneJev server and stops it when you switch models or close the platform. Select **Decision model** to switch between local OneJev and hosted GLM-5.3-Flash; set `OPENROUTER_API_KEY` in the environment to use GLM. The live view starts at Pokémon Square. Use **Load checkpoint** to restore a local `.state` or mGBA `.ss0` file; the agent pauses and clears its decision history. **Reset game** returns to the most recently loaded checkpoint. A goal change during inference takes effect after the current action. Append `--context mystery_dungeon/prompts/controls-v1.md` to the play command for the optional control guide. Other options include `--history-frames N` (default 1), `--completion-votes N` (default 1), and `--decision-model module:Class` for another no-argument model class.

To use hosted GLM-5.3-Flash through OpenRouter with `OPENROUTER_API_KEY` in a `.env` file, run `uv run --env-file .env -m mystery_dungeon play --decision-model mystery_dungeon.decision_models.glm:GlmDecisionModel`. This adapter sends screenshots to OpenRouter; it does not need the local OneJev server or the `onejev` extra.

## Benchmark and tests

```sh
uv run --env-file .env -m mystery_dungeon.benchmark --output runs/benchmark/town-example
uv run --extra onejev pytest -q
```

The benchmark uses the `square`, `face-lombre`, `face-npc`, `npc-dialogue`, and `shop-interact` checkpoints and defaults to GLM-5.3-Flash, so provide `OPENROUTER_API_KEY` through the environment or `.env` before running it. Its twelve cases progress from one move and leaving Pokémon Square through the north exit to approaching a visible Pokémon, starting and finishing dialogue, and using Wigglytuff's menus to open Friend Areas or check Pidgey. `face_npc_talk` opens Lombre's dialogue, `face_lombre_finish` completes it, and `face_npc_talk_overlap` retains the harder Granbull checkpoint with overlapping sprites. A hidden grader reads player coordinates, dialogue state, shop state, and the current map from game RAM to measure goal reach independently of the decision model. The RAM addresses apply to the matching Red Rescue Team ROM. Results are in `report.md` and `results.json`, with journals and screenshots under `cases/<case-id>/`. Use `--case <case-id>` for a single case or `--decision-model module:Class` to compare another model. For local OneJev runs, repeat `--model-file <path>` for weights and the vision projector to record their hashes. Earlier results from the retired `post-cutscene.ss0` benchmark are archived in [the historical comparison](docs/benchmark-vision-context.md).

Each benchmark case has an action budget and no wall-clock cutoff. Elapsed seconds are still reported.

## Code and data

- `emulator.py` wraps the Libretro core.
- `mystery_dungeon/agent.py` owns each gameplay turn: choose an action or stop, apply it, track the result, and record experience.
- `mystery_dungeon/session.py` provides emulator I/O: apply inputs and capture resulting frames.
- `mystery_dungeon/ram_state.py` reads field, dungeon, dialogue, and menu state plus the text printing flag, speaker species, and ground position. These offsets require the Red Rescue Team ROM with SHA-1 `9f4cfc5b5f4859d17169a485462e977c7aac2b89`. The agent waits for a dialogue page to finish before calling the decision model again; live `/api/state` exposes the latest `game_state`. A completed type-4 text box can be either dialogue or a menu, so its `mode` is only a best guess.
- `mystery_dungeon/decision_models/` contains adapters that score actions and visible goal completion. A model implements `score(goal, guidance, current_frame, recent_steps, button_options) -> DecisionScores`.
- `mystery_dungeon/live.py`, `play.py`, and `benchmark.py` are the runners; `scripts/` holds one-off experiments.

Generated data belongs in ignored `runs/`. A run contains `config.json`, `decisions.jsonl`, numbered `frames/`, `states/`, and snapshots of any loaded prompt or guide. OneJev loads `mystery_dungeon/prompts/onejev-v1.json`; set `QEV_PROMPT_VERSION=vN` to select another immutable version. Run configs record prompt, guide, and button-option hashes. Historical runs retain their earlier layout.

GLM decision and error entries in `decisions.jsonl` include `model_response` when OpenRouter returns JSON. It contains the returned reasoning, finish reason, and token usage when available; requests and API keys are not logged.

Further context: [game planning reference](docs/game-context.md), [one-step diagnosis](docs/one-step-diagnosis.md), and [historical vision benchmark](docs/benchmark-vision-context.md).
