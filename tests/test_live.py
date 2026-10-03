"""Small runnable check for live control and action recording."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from pezo.live import LiveRun


class FakeEmulator:
    def __init__(self, *args):
        self.buttons = []
        self.loaded_states = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def load_state(self, data):
        assert data == b"start"
        self.loaded_states.append(data)

    def step(self, frames=1, buttons=()):
        self.buttons.append(tuple(buttons))

    def screenshot(self):
        return self

    def png_bytes(self):
        return b"screenshot"

    def save_state(self):
        return b"checkpoint"


class LiveTest(unittest.TestCase):
    def test_step_and_manual_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "start.state"
            state.write_bytes(b"start")
            classifier = lambda image, goal, recent, options: {
                button: 1.0 if button == "A" else 0.0 for button in options
            }
            emulator = FakeEmulator()
            with patch("pezo.live.load_classifier", return_value=classifier), \
                 patch("pezo.live.GbaEmulator", return_value=emulator):
                run = LiveRun(root / "game.gba", root / "core", state, root / "output",
                              "fake:classify", 8, 4)
                thread = threading.Thread(target=run.run)
                thread.start()
                try:
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: run.loaded or run.error, 2))
                    run.control({"goal": "find exit", "steering": "check menu", "command": "step"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 1, 2))
                        self.assertFalse(run.running)
                    run.control({"command": "manual", "button": "RIGHT"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 2, 2))
                    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
                    self.assertEqual([row["choice"] for row in rows], ["A", "RIGHT"])
                    self.assertEqual(rows[0]["steering"], "check menu")
                    self.assertEqual(rows[1]["source"], "manual")
                    self.assertEqual((run.output / "latest.state").read_bytes(), b"checkpoint")
                    run.control({"command": "reset"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: run.reset_count == 1, 2))
                        self.assertEqual(run.frame, 1)
                        self.assertFalse(run.history)
                        self.assertFalse(run.running)
                        self.assertEqual(run.goal, "find exit")
                    self.assertEqual(emulator.loaded_states, [b"start", b"start"])
                    self.assertTrue((run.output / "before-reset-000002.state").is_file())
                    run.control({"command": "step"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 1, 2))
                        self.assertEqual(run.history[0]["step"], 1)
                    rows = [json.loads(line) for line in (run.output / "decisions.jsonl").read_text().splitlines()]
                    self.assertEqual(rows[2]["event"], "reset")
                    self.assertEqual(rows[3]["index"], 3)
                finally:
                    with run.condition:
                        run.stopping = True
                        run.condition.notify_all()
                    thread.join(2)
                    self.assertFalse(thread.is_alive())

    def test_new_direction_discards_inflight_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "start.state"
            state.write_bytes(b"start")
            started, release = threading.Event(), threading.Event()
            prompts = []

            def classifier(image, prompt, recent, options):
                prompts.append(prompt)
                if len(prompts) == 1:
                    started.set()
                    assert release.wait(2)
                return {button: 1.0 if button == "A" else 0.0 for button in options}

            with patch("pezo.live.load_classifier", return_value=classifier), \
                 patch("pezo.live.GbaEmulator", FakeEmulator):
                run = LiveRun(root / "game.gba", root / "core", state, root / "output",
                              "fake:classify", 8, 4)
                thread = threading.Thread(target=run.run)
                thread.start()
                try:
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: run.loaded or run.error, 2))
                    run.control({"goal": "old goal", "command": "step"})
                    self.assertTrue(started.wait(2))
                    run.control({"goal": "new goal", "steering": "go carefully"})
                    release.set()
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 1, 2))
                    self.assertEqual(len(prompts), 2)
                    self.assertIn("new goal", prompts[1])
                    self.assertEqual(run.snapshot()["history"][0]["goal"], "new goal")
                finally:
                    release.set()
                    with run.condition:
                        run.stopping = True
                        run.condition.notify_all()
                    thread.join(2)
                    self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
