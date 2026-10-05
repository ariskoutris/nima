"""Small runnable check for live control and action recording."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from mystery_dungeon.live import LiveRun
from mystery_dungeon.play import choose


class FakeEmulator:
    def __init__(self, *args):
        self.buttons = []
        self.loaded_states = []
        self.width = self.height = 16
        self.rgba = bytes(16 * 16 * 4)

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
    def test_live_pauses_when_same_button_has_no_visual_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "start.state"
            state.write_bytes(b"start")

            def classifier(image, goal, recent, options):
                return {button: 1.0 if button == "DOWN" else 0.0 for button in options}

            with patch("mystery_dungeon.live.load_classifier", return_value=classifier), \
                 patch("mystery_dungeon.live.GbaEmulator", FakeEmulator):
                run = LiveRun(root / "game.gba", root / "core", state, root / "output",
                              "fake:classify", 8, 4)
                thread = threading.Thread(target=run.run)
                thread.start()
                try:
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: run.loaded or run.error, 2))
                    run.control({"goal": "move down", "command": "start"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: bool(run.notice), 2))
                        self.assertFalse(run.running)
                        self.assertEqual(len(run.history), 8)
                        self.assertTrue(run.history[-1]["stalled"])
                finally:
                    with run.condition:
                        run.stopping = True
                        run.condition.notify_all()
                    thread.join(2)
                    self.assertFalse(thread.is_alive())

    def test_classifier_receives_full_goal_history(self):
        seen = []

        def classifier(image, goal, buttons, options):
            seen.extend(buttons)
            return {button: 1.0 if button == "RIGHT" else 0.0 for button in options}

        choose(classifier, Path("unused.png"), "move down five times", ["DOWN"] * 5)
        self.assertEqual(seen, ["DOWN"] * 5)

    def test_recent_buttons_toggle_independent_of_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "start.state"
            state.write_bytes(b"start")
            seen = []

            def classifier(image, goal, recent, options):
                seen.append((goal, recent))
                return {button: 1.0 if button == "DOWN" else 0.0 for button in options}

            with patch("mystery_dungeon.live.load_classifier", return_value=classifier), \
                 patch("mystery_dungeon.live.GbaEmulator", FakeEmulator):
                run = LiveRun(root / "game.gba", root / "core", state, root / "output",
                              "fake:classify", 8, 4)
                thread = threading.Thread(target=run.run)
                thread.start()
                try:
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: run.loaded or run.error, 2))
                    run.control({"goal": "move right", "command": "manual", "button": "RIGHT"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 1, 2))
                    run.control({"goal": "move down", "command": "step"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 2, 2))
                    self.assertEqual(seen[0], ("move down", ()))
                    run.control({"steering": "follow path", "use_recent": False, "command": "step"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 3, 2))
                    self.assertEqual(seen[1], ("move down\nOperator guidance: follow path", ()))
                    self.assertFalse(run.snapshot()["use_recent"])
                    run.control({"use_recent": True, "command": "step"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 4, 2))
                    self.assertEqual(seen[2], ("move down\nOperator guidance: follow path", ()))
                    run.control({"command": "step"})
                    with run.condition:
                        self.assertTrue(run.condition.wait_for(lambda: len(run.history) == 5, 2))
                    self.assertEqual(seen[3], ("move down\nOperator guidance: follow path", ("DOWN",)))
                    with self.assertRaisesRegex(ValueError, "use_recent must be a boolean"):
                        run.control({"use_recent": "false"})
                finally:
                    with run.condition:
                        run.stopping = True
                        run.condition.notify_all()
                    thread.join(2)
                    self.assertFalse(thread.is_alive())

    def test_step_and_manual_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "start.state"
            state.write_bytes(b"start")
            classifier = lambda image, goal, recent, options: {
                button: 1.0 if button == "A" else 0.0 for button in options
            }
            emulator = FakeEmulator()
            with patch("mystery_dungeon.live.load_classifier", return_value=classifier), \
                 patch("mystery_dungeon.live.GbaEmulator", return_value=emulator):
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

            with patch("mystery_dungeon.live.load_classifier", return_value=classifier), \
                 patch("mystery_dungeon.live.GbaEmulator", FakeEmulator):
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
