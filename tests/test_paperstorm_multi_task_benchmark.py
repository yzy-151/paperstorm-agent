import tempfile
import unittest
from pathlib import Path


class PaperStormMultiTaskBenchmarkTest(unittest.TestCase):
    def test_task_groups_are_defined(self):
        from knowledge_storm.paperstorm_multi_task_benchmark import TASK_GROUPS

        self.assertGreaterEqual(len(TASK_GROUPS), 4)
        names = {group["name"] for group in TASK_GROUPS}
        self.assertEqual(len(names), len(TASK_GROUPS))
        for group in TASK_GROUPS:
            self.assertTrue(group["label"])
            self.assertTrue(group["terms"])

    def test_compare_stacks_shows_v41_improvement_on_seed(self):
        from knowledge_storm.paperstorm_eval_v4 import build_seed_dataset
        from knowledge_storm.paperstorm_multi_task_benchmark import compare_stacks_on_dataset
        from knowledge_storm.paperstorm_rag import HashEmbeddingProvider

        comparison = compare_stacks_on_dataset(
            build_seed_dataset(),
            embedding_provider=HashEmbeddingProvider(64),
            top_k=5,
        )
        self.assertGreater(
            comparison["v41"]["recall_at_k"],
            comparison["legacy"]["recall_at_k"],
        )
        self.assertGreater(comparison["deltas"]["relative_recall_gain_pct"], 0)


if __name__ == "__main__":
    unittest.main()
