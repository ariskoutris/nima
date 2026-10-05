"""Exercise benchmark grading and stops against the actual checkpoint."""

import tempfile
import unittest
from pathlib import Path

from emulator import GbaEmulator
from mystery_dungeon.benchmark import (CASES, CORE, ROM, STATE, MAILBOX_ROUTE, Case, Grader,
                                       build_fixtures, read_ss0, restore, run_case)


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(path.is_file() for path in (ROM, CORE, STATE)):
            raise unittest.SkipTest("local ROM, core, and mGBA checkpoint required")
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.game = GbaEmulator(ROM, CORE)
        cls.game.__enter__()
        cls.source = read_ss0(STATE)
        cls.seeds, cls.refs = build_fixtures(cls.game, cls.source, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.game.__exit__(None, None, None)
        cls.directory.cleanup()

    def test_checkpoint_contains_state_and_save_ram(self):
        self.assertEqual(len(self.source.save_ram), 131072)
        self.assertGreater(len(self.source.state), 300000)
        damaged = bytearray(STATE.read_bytes())
        damaged[100] ^= 1
        path = self.root / "damaged.ss0"
        path.write_bytes(damaged)
        with self.assertRaises(ValueError):
            read_ss0(path)

    def test_mailbox_side_and_npc_are_not_mailbox_success(self):
        restore(self.game, self.source)
        ram = self.game.system_ram()
        grader = Grader(self.refs, ram)
        for name in ("outside", "mailbox_side", "npc"):
            self.assertFalse(grader.observe(self.refs[name], ram, "mailbox")[0], name)
        self.assertTrue(grader.observe(self.refs["mailbox"], ram, "mailbox")[0])
        self.assertEqual(grader.observe(self.refs["cave"], ram, "mailbox_cave")[0], True)

    def test_fixed_button_policy_succeeds_and_wait_loop_stops(self):
        def always(button):
            return lambda image, goal, recent, options: {
                option: 1.0 if option == button else 0.0 for option in options
            }

        south = next(case for case in CASES if case.id == "room_south")
        result = run_case(self.game, south, "direct", always("DOWN"), None,
                          self.seeds, self.refs, self.root / "south")
        self.assertEqual(result["stop_reason"], "success")
        self.assertGreater(result["steps"], 0)

        blocked = Case("wait_loop", "outside", "Find the clearing", "clearing", 16)
        result = run_case(self.game, blocked, "direct", always("WAIT"), None,
                          self.seeds, self.refs, self.root / "wait_loop")
        self.assertIn(result["stop_reason"], ("loop", "stalled", "budget"))
        self.assertFalse(result["success"])

    def test_every_case_has_a_successful_reference_route(self):
        routes = {
            "room_south": ["DOWN"],
            "room_exit_directional": ["DOWN"] * 8,
            "room_exit_goal": ["DOWN"] * 8,
            "outside_talk": ["DOWN", "A"],
            "npc_ready_talk": ["A"],
            "outside_mailbox": list(MAILBOX_ROUTE),
            "mailbox_front_check": ["A"],
            "mailbox_front_button": ["A"],
            "room_then_talk": ["DOWN"] * 9 + ["A"],
            "outside_clearing": ["RIGHT"] * 7 + ["UP"] * 10,
            "bridge_to_clearing": ["UP"] * 10,
            "bridge_up_literal": ["UP"] * 10,
            "mailbox_to_cave": ["A"] * 120,
            "outside_mailbox_to_cave": list(MAILBOX_ROUTE) + ["A"] * 140,
        }
        for case in CASES:
            route = routes[case.id]

            def scripted(image, goal, recent, options):
                button = route[min(len(recent), len(route) - 1)]
                return {name: float(name == button) for name in options}

            with self.subTest(case=case.id):
                result = run_case(self.game, case, "direct", scripted, None,
                                  self.seeds, self.refs, self.root / f"reference_{case.id}")
                self.assertTrue(result["success"], result)


if __name__ == "__main__":
    unittest.main()
