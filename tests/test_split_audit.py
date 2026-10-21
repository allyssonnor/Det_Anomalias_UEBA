import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.data.dataset_adapter import DatasetAdapter


class SplitAuditTests(unittest.TestCase):
    def test_random_split_audit_records_prevalence_and_shared_identities(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = {
                "output_dir": output_dir,
                "data": {"target_col": "Is_Anomaly"},
                "sampling": {
                    "split_strategy": "random",
                    "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
                },
                "training": {"seed": 42},
            }
            frame = pd.DataFrame({
                "UserID": [f"user_{i % 4}" for i in range(100)],
                "AttackID": [f"attack_{i % 5}" for i in range(100)],
                "Time": list(range(100)),
                "Is_Anomaly": [int(i % 10 == 0) for i in range(100)],
            })

            DatasetAdapter(config)._perform_dynamic_split(frame)

            audit_path = Path(output_dir) / "split_audit_random_seed42.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(audit["partitions"]["train"]["events"], 70)
        self.assertEqual(audit["partitions"]["validation"]["events"], 20)
        self.assertEqual(audit["partitions"]["test"]["events"], 10)
        self.assertGreater(audit["overlap"]["train_validation"]["UserID"], 0)
        self.assertGreater(audit["overlap"]["train_test"]["AttackID"], 0)


if __name__ == "__main__":
    unittest.main()
