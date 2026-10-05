"""Keep action history out of the small vision model's button prompt."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mystery_dungeon.classifiers.onejev import classify
from mystery_dungeon.play import with_context


class OneJevTest(unittest.TestCase):
    def test_general_controls_are_sent_on_every_decision(self):
        calls = []

        def classifier(image, goal, recent, options):
            calls.append(goal)
            return {name: 0.0 for name in options}

        wrapped = with_context(classifier, "D-pad moves the character. A confirms.")
        wrapped(Path("screen.png"), "Find the exit", (), {"A": "confirm"})
        wrapped(Path("screen.png"), "Talk to the NPC", (), {"A": "confirm"})
        self.assertEqual(calls, [
            "D-pad moves the character. A confirms.\n\nCurrent goal: Find the exit",
            "D-pad moves the character. A confirms.\n\nCurrent goal: Talk to the NPC",
        ])
        self.assertIs(with_context(classifier, ""), classifier)

    def test_repeated_buttons_are_not_sent_as_context(self):
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "screen.png"
            screenshot.write_bytes(b"image")
            response = SimpleNamespace(answers={"next": SimpleNamespace(probabilities={"UP": 1.0})})
            with patch("mystery_dungeon.classifiers.onejev.Client") as client:
                client.return_value.system_one.return_value = response
                classify(screenshot, "Move up", ("LEFT",) * 8, {"UP": "press up"})

            self.assertEqual(client.return_value.system_one.call_args.kwargs["state"],
                             {"screen": "<image:1>", "task": {"instruction": "Move up"}})


if __name__ == "__main__":
    unittest.main()
