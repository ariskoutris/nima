"""Start the configured model server or a live game session."""

import os
import sys
import tomllib
from pathlib import Path


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command not in {"serve", "play"}:
        raise SystemExit("usage: python -m mystery_dungeon {serve|play} [options]")
    config = tomllib.loads((Path(__file__).parents[1] / "config.toml").read_text())
    if command == "serve":
        server = config["server"]
        args = ["-m", "qev.cli", "serve", "--gguf", server["gguf"],
                "--mmproj", server["mmproj"], *sys.argv[2:]]
    else:
        game = config["game"]
        args = ["-m", "mystery_dungeon.live", "--core", game["core"],
                "--state", game["state"], *sys.argv[2:], game["rom"]]
    os.execv(sys.executable, [sys.executable, *args])


if __name__ == "__main__":
    main()
