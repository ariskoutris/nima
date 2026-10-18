"""Serve a configured model or start a live game session."""

import argparse
import os
import sys
import tomllib

from mystery_dungeon.models import CONFIG, read_models
from mystery_dungeon.models.server import serve_command


def serve(argv: list[str]) -> None:
    """Replace this process with a model's serve command."""
    models = read_models()
    parser = argparse.ArgumentParser(prog="python -m mystery_dungeon serve",
                                     description="Run a model's serve command in the foreground.")
    parser.add_argument("model", choices=models)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    entry = models[args.model]
    if not entry.serve:
        parser.error(f"models.{args.model} has no serve command")
    command = serve_command(entry, args.port)
    os.chdir(CONFIG.parent)  # relative paths in serve commands are relative to config.toml
    os.execvp(command[0], command)


def play(argv: list[str]) -> None:
    """Replace this process with the live view over the configured game."""
    game = tomllib.loads(CONFIG.read_text())["game"]
    args = ["-m", "mystery_dungeon.live", "--core", game["core"],
            "--state", game["state"], *argv, game["rom"]]
    os.execv(sys.executable, [sys.executable, *args])


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    match command:
        case "serve":
            serve(sys.argv[2:])
        case "play":
            play(sys.argv[2:])
        case _:
            raise SystemExit("usage: python -m mystery_dungeon {serve|play} [options]")


if __name__ == "__main__":
    main()
