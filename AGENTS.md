# mystery-dungeon

Vision agent for Pokémon Mystery Dungeon: Red Rescue Team. A solo research project: favour clear
experiments and deletion over backwards compatibility.

## Commands
- Test: `uv run python -m pytest -q`
- Lint: `uv run ruff check` (`--fix` sorts imports)
- Play: `uv run --env-file .env -m mystery_dungeon.live`
- Benchmark: `uv run --env-file .env -m mystery_dungeon.benchmark [--case ID]`
- Probe: `uv run --env-file .env -m scripts.probe [--case ID] [--variant NAME]`
- `--env-file .env` supplies the API key named by `[model] api_key_env` in `config.toml`.
- Don't start benchmarks, probes or model runs unless asked; give the command instead.

## Architecture
- The agent observes the game only through frames and acts only through actions. It has no
  access to RAM or any other game state.
- `GameplaySession` is the only code that interacts with the game: it executes inputs, captures
  observations and is the only reader of RAM (`session.ram_state()`).
- One accepted exception: the session uses RAM to time captures. It waits until text finishes
  printing and returns menu frames with the cursor visible. It never skips cutscenes or advances
  dialogue; the agent chooses WAIT.
- RAM state from the session may go to the operator (live view) and to `benchmark.py` for grading,
  never to the agent or its model.
- Game handling covers a kind of behaviour (all text boxes, all menus), never one screen or game
  state; don't write a handler per case. Name measured timings as constants and say in a comment
  what was measured.

## Scope
- Do what was asked. Mention follow-ups (README, flags, compatibility) in one line; don't add them
  to the plan.
- When the user calls a decision final, delete what it replaces: no versioned files, no flags to
  restore old behaviour.
- Experiment settings live in `scripts/probe.py`. Production code gets a setting only once the user
  adopts it. When an experiment ends, record question → result → decision in `docs/` and delete its
  code.
- On a correction, change only what it names. If the scope is unclear, restate what changes and
  what stays first.
- Back claims with evidence ("measured on the shop menu and Pokémon list"), not "guaranteed".

## Object design
- Give each module and class one responsibility: the agent decides, the session drives the emulator, runners own logging and run output, the live server coordinates operator input. Call another object's public methods instead of reaching into its state.
- Add a Protocol only at a seam with two or more real implementations (e.g. the chat model plus scripted test models). For a single implementation, use the concrete class; no ABCs, factories or registries.
- Compose objects instead of inheriting. To share code, write a module-level function, not a base class.
- Use `@dataclass(frozen=True)` for values passed between components. Only the owning class mutates its state.
- Model closed sets as type unions dispatched with `match`, so adding a variant means one new dataclass plus one case.
- Implement the full protocol in every implementation, including test doubles.
- Fail loudly: raise on invalid input or model output and convert exceptions to an error result in one place. Never substitute a silent default.
- Keep side effects at the edges: the model call sees only the context and frame and never touches the game; the agent applies the action after it returns, and runners own files and logs.

## Names
One term per concept in code, prompts, log fields, flags and docs:
- task: what the agent is currently pursuing; the model decides when it is complete by
  answering DONE instead of an action.
- task list: a tree of tasks rooted at "beat the game". A task's children are prerequisites that
  must be completed first (e.g. level up before a boss). The agent maintains it because what to
  attempt next isn't always clear. The design is pending; this core idea is fixed.
- action: one `Press(button)` or `Wait()`, the closed set in `actions.py`.
- short-term memory: recent reasoning, observations and turns, plus information retrieved for the
  current task.
- long-term memory (knowledge): a journal of observations, experiences and information gathered
  during play. The agent draws on it to plan new tasks, judge which are feasible and prioritize;
  task information comes from here. It starts with prerequisite information such as
  `knowledge/controls.md`. Not "guide". The memory design is pending; this two-level
  split is fixed.
- model: what the agent asks each turn to choose an action or DONE (`Model`, `ChatModel`),
  configured by `[model]` in `config.toml`; `name` is its endpoint id.
- context: what the model acts on each turn besides the current frame: the task, knowledge and
  short-term memory (`agent.Context`).
- frame: an image in memory. screenshot: its saved PNG.
- case: one benchmark task. variant: one probe configuration. limit: a case's action budget.
- Don't use "goal" or "decision". "steering" is reserved for future mid-run operator guidance.

## Documentation
- Start every module with a one-line docstring stating its responsibility.
- Write a docstring on a public class or function when its contract isn't obvious from the name and types. Say what it does and guarantees, not how. Add `Raises:` when callers must handle an exception; use Google-style `Args:`/`Returns:` only for non-obvious parameters.
- Write comments that explain why (game quirks, timing, constraints). Delete comments that restate the code, and match the surrounding comment density.
- Delete removed concepts completely: no commented-out code, compatibility aliases or stale names in code, prompts, logs, tests or docs.
- README.md covers setup and the commands it lists; change it only when those change.
- `docs/` holds dated experiment notes and is not updated afterwards.

## General practice
- Type-hint every public signature. Put constants in UPPER_CASE at module top.
- Define each path, hash, timing and model setting once: paths and model settings in
  `config.toml`, the rest as a constant in its owning module.
- Keep functions small with early returns. Replace boolean parameters that switch behaviour with two functions or an explicit value.
- Prefer the standard library; ask before adding a dependency.
- Test behaviour through public APIs, never private attributes. Every bug fix or new rule gets a test.

## Commits
- Imperative, sentence-case subject with no type prefix; the body explains why. One concern per
  commit.
- Never include unrelated pre-existing changes; list them instead.
- Before committing a non-trivial change, run one review (`/code-review` in Claude Code).
