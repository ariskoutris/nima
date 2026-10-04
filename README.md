# mystery-dungeon

A vision-driven agent prototype for Pokémon Mystery Dungeon: Red Rescue Team (GBA). mGBA supplies screenshots and executes button presses; OneJev is the current decision backend.

## Run

Requires `uv`, `llama-server`, an mGBA Libretro core at `.local/cores/mgba_libretro.dylib`, a ROM in `roms/`, and a checkpoint at `.local/baselines/post_continue.state`.

Start the model server and live view in separate terminals:

```sh
HF_HOME="$PWD/.local/hf-cache" uv run --extra onejev qev serve \
  --gguf .local/models/onejev-0.8b-q4_k_m.gguf \
  --mmproj .local/models/onejev-0.8b-mmproj-q8_0.gguf
```

```sh
uv run --extra onejev python -m mystery_dungeon.live \
  'roms/Pokemon Mystery Dungeon - Red Rescue Team (USA, Australia).gba'
```

The live view runs at http://127.0.0.1:8765. Sessions are saved under `.local/live/`. Pass `--classifier module:callable` to use another screenshot classifier.
