import tempfile
import unittest

import pandas as pd

from scripts.evaluate_performance import run_external_evaluation


class PrevalenceMetricsTests(unittest.TestCase):
    def test_evaluation_reports_prevalence_and_normalized_pr_auc(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = {
                "output_dir": output_dir,
                "data": {"target_col": "Is_Anomaly"},
                "model": {"sequence_length": 1},
                "training": {"threshold_mode": "global"},
            }
            frame = pd.DataFrame({
                "UserID": ["u", "u"],
                "Time": [1, 2],
                "Is_Anomaly": [0, 1],
            })

            metrics = run_external_evaluation(
                config,
                test_df=frame,
                scores=[0.1, 0.9],
                indices=[0, 1],
            )

        self.assertEqual(metrics["prevalence"], 0.5)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["lift_pr"], 2.0)


if __name__ == "__main__":
    unittest.main()
