import unittest

from readmission.metrics import average_precision, compute_metrics, select_threshold


class MetricsTests(unittest.TestCase):
    def test_metrics_fail_clearly_for_one_class_split(self):
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            average_precision([0, 0], [0.1, 0.2])

    def test_threshold_is_validation_selected(self):
        labels = [0, 1, 1, 0]
        scores = [0.1, 0.8, 0.7, 0.6]
        threshold = select_threshold(labels, scores)
        metrics = compute_metrics(labels, scores, threshold)
        self.assertEqual(threshold, 0.7)
        self.assertEqual(metrics.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
