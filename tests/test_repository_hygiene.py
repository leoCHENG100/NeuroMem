"""Check artifact exclusion with Git's ignore matcher in a temporary repository."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "Git is required to check ignore rules")
class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name)
        subprocess.run(["git", "init", "--quiet", str(self.repository)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (self.repository / ".gitignore").write_text((ROOT / ".gitignore").read_text(encoding="utf-8"),
                                                   encoding="utf-8")

    def is_ignored(self, path):
        result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", "--", path],
                                cwd=self.repository, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertIn(result.returncode, (0, 1), result.stderr.decode())
        return result.returncode == 0

    def test_prediction_artifacts_are_ignored_at_any_output_location(self):
        for path in (
            "00000.prediction.txt", "example-run/00000.prediction.txt",
            "example-run/predictions/00000.prediction.txt", "predictions/custom-name.txt",
            "nested/example-run/predictions/custom-name.txt", "example-run/task_records.jsonl",
            "example-run/policy_state.pkl",
        ):
            with self.subTest(path=path):
                self.assertTrue(self.is_ignored(path))

    def test_source_documentation_and_input_checksums_remain_publishable(self):
        for path in (
            ".gitignore", "README.md", "neuromem_policy.py", "tests/test_repository_hygiene.py",
            "checksums/mimic_iii.sha256", "docs/prediction-format.md",
        ):
            with self.subTest(path=path):
                self.assertFalse(self.is_ignored(path))
