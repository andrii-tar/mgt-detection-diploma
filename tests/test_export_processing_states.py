from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class ExportProcessingStatesTest(unittest.TestCase):
    def _load_exporter(self, data_root: Path):
        previous_data_root = os.environ.get("TEXT_DETECTION_DATA")
        os.environ["TEXT_DETECTION_DATA"] = str(data_root)
        try:
            import monitoring.scripts.export_processing_states as exporter

            return importlib.reload(exporter)
        finally:
            if previous_data_root is None:
                os.environ.pop("TEXT_DETECTION_DATA", None)
            else:
                os.environ["TEXT_DETECTION_DATA"] = previous_data_root

    def test_try_write_states_preserves_existing_output_on_filesystem_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exporter = self._load_exporter(root)
            output = root / "processing-states.json"
            output.write_text('[{"dataset": "old"}]', encoding="utf-8")

            with mock.patch.object(exporter, "collect_states", side_effect=OSError(5, "Input/output error")):
                ok = exporter.try_write_states(root / "processed", output)

            self.assertFalse(ok)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [{"dataset": "old"}])

    def test_try_write_states_returns_true_and_writes_states_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exporter = self._load_exporter(root)
            output = root / "processing-states.json"

            with mock.patch.object(exporter, "collect_states", return_value=[{"dataset": "new"}]):
                ok = exporter.try_write_states(root / "processed", output)

            self.assertTrue(ok)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [{"dataset": "new"}])

    def test_dataset_choices_are_sorted_from_configured_datasets(self) -> None:
        exporter = self._load_exporter(Path.cwd())
        with mock.patch.object(exporter, "DATASETS", {"zeta": {}, "alpha": {}}):
            self.assertEqual(exporter.dataset_choices(), ["alpha", "zeta"])

    def test_write_dataset_choices_writes_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            exporter = self._load_exporter(root)
            output = root / "datasets.json"

            with mock.patch.object(exporter, "DATASETS", {"zeta": {}, "alpha": {}}):
                exporter.write_dataset_choices(output)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), ["alpha", "zeta"])


if __name__ == "__main__":
    unittest.main()
