import unittest


class DenseScaleBenchmarkTests(unittest.TestCase):
    def test_flat_memory_estimate_is_dimension_aware(self):
        from examples.storm_examples.benchmark_dense_scale import estimate_flat_bytes

        self.assertEqual(8_192_000_000, estimate_flat_bytes(2_000_000, 1024))

    def test_scale_report_separates_measurement_from_estimate(self):
        from examples.storm_examples.benchmark_dense_scale import build_scale_report

        report = build_scale_report(
            measured={"vector_count": 100_000, "dimension": 384},
            estimated_count=2_000_000,
        )

        self.assertEqual(100_000, report["measured"]["vector_count"])
        self.assertEqual(2_000_000, report["estimated"]["vector_count"])
        self.assertEqual("estimate", report["estimated"]["evidence_tier"])
        self.assertNotIn("query_p95_ms", report["estimated"])


if __name__ == "__main__":
    unittest.main()
