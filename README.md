# mystery-dungeon

A vision-driven agent prototype for Pokémon Mystery Dungeon: Red Rescue Team (GBA). mGBA supplies screenshots and executes button presses; OneJev is the current decision backend.

## Run

Requires `uv`, `llama-server`, an mGBA Libretro core, a GBA ROM, and a Libretro checkpoint.

Start the model server and live view in separate terminals:

```sh
uv run --extra onejev qev serve \
  --gguf data/engines/models/onejev-0.8b-q4_k_m.gguf \
  --mmproj data/engines/models/onejev-0.8b-mmproj-q8_0.gguf
```

```sh
uv run --extra onejev python -m mystery_dungeon.live \
  --core data/engines/cores/mgba_libretro.dylib \
  --state data/game/checkpoints/post_continue.state \
  data/game/roms/red-rescue-team.gba
```

The live view runs at http://127.0.0.1:8765. Sessions are saved under `runs/`. Pass `--classifier module:callable` to use another screenshot classifier.
