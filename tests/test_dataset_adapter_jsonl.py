import json
import tempfile
import unittest
from pathlib import Path

from core.data.dataset_adapter import DatasetAdapter


class JsonlContentDetectionTests(unittest.TestCase):
    def test_jsonl_content_is_detected_even_when_extension_is_txt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.txt"
            path.write_text(
                json.dumps({"UserID": "u1", "Time": 10, "Is_Anomaly": 0}) + "\n"
                + json.dumps({"UserID": "u2", "Time": 20, "Is_Anomaly": 1}) + "\n",
                encoding="utf-8",
            )
            adapter = DatasetAdapter({"sampling": {"mode": "full"}})

            frame = adapter._load_resource(str(path))

        self.assertEqual(frame["UserID"].tolist(), ["u1", "u2"])
        self.assertEqual(frame["Time"].tolist(), [10, 20])
        self.assertEqual(frame["Is_Anomaly"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
