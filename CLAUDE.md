# Conventions

## Object design
- Give each module and class one responsibility: the agent decides, the session drives the emulator, runners own logging and run output, the live server coordinates operator input. Call another object's public methods instead of reaching into its state.
- Add a Protocol only at a seam with two or more real implementations (e.g. model adapters plus scripted test models). For a single implementation, use the concrete class; no ABCs, factories or registries.
- Compose objects instead of inheriting. To share code, write a module-level function, not a base class.
- Use `@dataclass(frozen=True)` for values passed between components. Only the owning class mutates its state.
- Model closed sets as type unions dispatched with `match`, so adding a variant means one new dataclass plus one case.
- Implement the full protocol in every implementation, including test doubles.
- Fail loudly: raise on invalid input or model output and convert exceptions to an error result in one place. Never substitute a silent default.
- Keep side effects at the edges: deciding what to do must not touch the game, files or network; a separate step applies it.

## Documentation
- Start every module with a one-line docstring stating its responsibility.
- Write a docstring on a public class or function when its contract isn't obvious from the name and types. Say what it does and guarantees, not how. Add `Raises:` when callers must handle an exception; use Google-style `Args:`/`Returns:` only for non-obvious parameters.
- Write comments that explain why (game quirks, timing, constraints). Delete comments that restate the code, and match the surrounding comment density.
- Delete removed concepts completely: no commented-out code, compatibility aliases or stale names in code, prompts, logs, tests or docs.
- Use one name per concept everywhere (code, prompts, log fields, README). Rename all occurrences together.
- Update README.md in the same change that alters flags, file formats or the model contract. Treat `docs/` as historical notes.

## General practice
- Type-hint every public signature. Put constants in UPPER_CASE at module top.
- Keep functions small with early returns. Replace boolean parameters that switch behaviour with two functions or an explicit value.
- Prefer the standard library; ask before adding a dependency.
- Test behaviour through public APIs, never private attributes. Every bug fix or new rule gets a test.
- Run tests with `uv run --extra onejev python -m pytest -q`.
- After tests pass on a non-trivial change, run `/simplify`, `/code-review` and `/ponytail-review` on the diff.
