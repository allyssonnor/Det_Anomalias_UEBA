import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.features.feature_processor import FeatureProcessor
from core.trainer import Trainer


class CaptureModel:
    def __init__(self):
        self.fit_X = None

    def fit(self, X):
        self.fit_X = np.asarray(X).copy()

    def score(self, X):
        return np.zeros(len(X), dtype=float)


class EarlyStoppingProtocolTests(unittest.TestCase):
    def test_training_windows_are_benign_contiguous_training_only(self):
        model = CaptureModel()
        config = {
            "data": {"target_col": "Is_Anomaly"},
            "model": {"sequence_length": 2},
            "training": {
                "supervised": False,
                "save_model": False,
                "resume": False,
                "threshold_mode": "global",
            },
        }
        train = pd.DataFrame({
            "UserID": ["u"] * 8,
            "Time": list(range(8)),
            "Is_Anomaly": [0, 1, 0, 0, 1, 0, 0, 0],
            "signal": [0.1, 99.0, 0.2, 0.3, 99.0, 0.4, 0.5, 0.6],
        })
        validation = pd.DataFrame({
            "UserID": ["u"] * 6,
            "Time": list(range(8, 14)),
            "Is_Anomaly": [0, 1, 0, 0, 1, 0],
            "signal": [0.2, 777.0, 0.3, 0.4, 777.0, 0.1],
        })

        with tempfile.TemporaryDirectory() as output_dir:
            config["output_dir"] = output_dir
            Trainer(config, model).run(train, validation, test_df=None)

        # Only windows [2,3], [5,6], [6,7] are all benign in original order.
        scaler = StandardScaler().fit(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(-1, 1))
        expected = np.stack([
            scaler.transform(np.array([[0.2], [0.3]])),
            scaler.transform(np.array([[0.4], [0.5]])),
            scaler.transform(np.array([[0.5], [0.6]])),
        ])
        np.testing.assert_allclose(model.fit_X, expected, rtol=1e-6, atol=1e-6)

    def test_logontype_ablation_removes_raw_and_derived_features(self):
        excluded = [
            "LogonType", "network_logon", "network_logon_rate",
            "rdp_event", "rdp_rate",
        ]
        processor = FeatureProcessor({
            "data": {"target_col": "Is_Anomaly"},
            "features": {"enabled": False, "exclude_features": excluded},
        })
        processor.fit(pd.DataFrame({
            "UserID": ["u"], "Time": [1], "Is_Anomaly": [0],
        }))
        frame = pd.DataFrame({
            "UserID": ["u"], "Time": [2], "Is_Anomaly": [0],
            "LogonType": [10], "network_logon": [1],
            "network_logon_rate": [0.8], "rdp_event": [1], "rdp_rate": [0.8],
            "benign_control": [0.2],
        })
        result = processor.transform(frame)

        self.assertTrue(set(excluded).isdisjoint(processor.final_feature_names))
        self.assertIn("benign_control", processor.final_feature_names)
        self.assertTrue(set(excluded).isdisjoint(result.columns))


if __name__ == "__main__":
    unittest.main()
