"""Run the game replay CLI with OneJev."""

import argparse
import os
import sys

from pezo.play import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--url")
    args, remaining = parser.parse_known_args(sys.argv[1:])
    if args.url:
        os.environ["QEV_BASE_URL"] = args.url
    main(remaining, default_classifier="pezo.classifiers.onejev:classify")
