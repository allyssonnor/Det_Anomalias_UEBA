import tempfile
import unittest

import numpy as np
import pandas as pd

from core.features.feature_processor import FeatureProcessor
from core.trainer import Trainer


class _SupervisedModel:
    def __init__(self):
        self.fit_called = False

    def fit(self, X, y=None):
        self.fit_called = True

    def score(self, X):
        return np.zeros(len(X))


class ProtocolSafeguardsTests(unittest.TestCase):
    def test_rolling_rates_forget_events_outside_window(self):
        processor = FeatureProcessor({"features": {"rate_window": 2}})
        frame = pd.DataFrame({"UserID": ["u", "u", "u"], "flag": [1, 0, 0]})

        rates = processor._rolling_rate(frame, "flag")

        np.testing.assert_allclose(rates.to_numpy(), [1.0, 0.5, 0.0])

    def test_logontype_can_be_removed_from_final_feature_vector(self):
        processor = FeatureProcessor({
            "features": {"enabled": False, "exclude_features": ["LogonType"]}
        })
        train = pd.DataFrame({
            "UserID": ["u"], "Time": [1], "Is_Anomaly": [0]
        })
        processor.fit(train)
        frame = pd.DataFrame({
            "UserID": ["u"], "Time": [2], "Is_Anomaly": [0],
            "LogonType": [10], "fail_rate": [0.2]
        })

        result = processor.transform(frame)

        self.assertNotIn("LogonType", processor.final_feature_names)
        self.assertIn("fail_rate", processor.final_feature_names)
        self.assertNotIn("LogonType", result.columns)

    def test_validation_attacks_are_not_used_to_fill_training_classes(self):
        model = _SupervisedModel()
        config = {
            "data": {"target_col": "Is_Anomaly"},
            "model": {"sequence_length": 1},
            "training": {"supervised": True, "save_model": False,
                         "threshold_mode": "global"},
        }
        train = pd.DataFrame({
            "UserID": ["u"] * 4, "Time": [1, 2, 3, 4],
            "Is_Anomaly": [0, 0, 0, 0], "signal": [0.1, 0.2, 0.3, 0.4],
        })
        validation = pd.DataFrame({
            "UserID": ["u"] * 4, "Time": [5, 6, 7, 8],
            "Is_Anomaly": [0, 1, 1, 0], "signal": [0.1, 0.9, 0.8, 0.2],
        })

        with tempfile.TemporaryDirectory() as output_dir:
            config["output_dir"] = output_dir
            trainer = Trainer(config, model)
            with self.assertRaisesRegex(ValueError, "Não use a validação"):
                trainer.run(train_df=train, val_df=validation)
        self.assertFalse(model.fit_called)


if __name__ == "__main__":
    unittest.main()
