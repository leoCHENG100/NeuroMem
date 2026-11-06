"""Keep the documented configuration and local links consistent with the code."""

from pathlib import Path
import re
import unittest

from neuromem_core import BudgetedSupportPathSelector, HashRelationScorer
from neuromem_policy import NeuroMemPolicy


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_main_configuration_matches_runtime_defaults(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        rows = re.findall(
            r"^\|[^|]+\|\s*([0-9.]+)\s*\|\s*`([a-z_]+)`\s*\|$",
            readme, re.MULTILINE,
        )
        documented = {setting: float(value) for value, setting in rows}
        scorer = HashRelationScorer()
        selector = BudgetedSupportPathSelector(scorer)
        policy = NeuroMemPolicy(scorer)
        owners = {
            "expansion_budget": selector,
            "candidate_pool_size": selector,
            "credit_horizon": policy,
            "pending_retrieval_threshold": policy,
            "promotion_score_threshold": policy,
            "promotion_min_utility": policy,
            "review_delay": policy,
            "max_pending_age": policy,
            "lambda_memory": policy,
            "beta_redundancy": policy,
        }
        self.assertEqual(set(documented), set(owners))
        for setting, owner in owners.items():
            with self.subTest(setting=setting):
                self.assertEqual(documented[setting], getattr(owner, setting))

    def test_documented_local_links_exist(self):
        for name in ("README.md", "NOTICE.md"):
            content = (ROOT / name).read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
                if "://" in target or target.startswith("#"):
                    continue
                path = ROOT / target.split("#", 1)[0]
                with self.subTest(document=name, target=target):
                    self.assertTrue(path.is_file())

    def test_readme_distinguishes_preflight_policy_state_and_full_stream_checks(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for contract in (
            "does not parse CSV contents", "CLI has no resume option",
            "same frozen relation scorer", "full-horizon policy updates during the task stream",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, readme)
