"""Keep the CLI smoke's learning checks independent of end-of-stream flushing."""

import unittest

from neuromem_core import HashRelationScorer
from neuromem_policy import NeuroMemPolicy
from tests.smoke_cli import SMOKE_TASK_COUNT, check_delayed_updates


class SmokeCoverageTests(unittest.TestCase):
    def test_task_stream_reaches_the_default_credit_deadline(self):
        policy = NeuroMemPolicy(HashRelationScorer())
        self.assertGreater(SMOKE_TASK_COUNT, policy.review_delay + policy.credit_horizon)

    def test_flush_only_updates_do_not_satisfy_stream_coverage(self):
        rows = [{"adam_step": 0, "finalized_decisions": []} for _ in range(15)]
        with self.assertRaisesRegex(AssertionError, "during the task stream"):
            check_delayed_updates(rows, {"open_decisions": 0, "adam_step": 3}, 32)

    def test_stream_and_flush_updates_are_checked_separately(self):
        rows = [{"adam_step": 1, "finalized_decisions": [{"elapsed": 32}]}]
        final = {"open_decisions": 0, "adam_step": 2}
        self.assertEqual(check_delayed_updates(rows, final, 32),
                         {"stream_updates": 1, "updates_after_flush": 2})
        with self.assertRaisesRegex(AssertionError, "final flush"):
            check_delayed_updates(rows, {"open_decisions": 1, "adam_step": 2}, 32)
        with self.assertRaisesRegex(AssertionError, "final flush"):
            check_delayed_updates(rows, {"open_decisions": 0, "adam_step": 1}, 32)
        rows[0]["finalized_decisions"][0]["elapsed"] = 31
        with self.assertRaisesRegex(AssertionError, "before its credit horizon"):
            check_delayed_updates(rows, final, 32)
