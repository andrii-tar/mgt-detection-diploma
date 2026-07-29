from __future__ import annotations

import unittest
import json
import os
from pathlib import Path

os.environ.setdefault("TEXT_DETECTION_DATA", str(Path(__file__).resolve().parents[1]))

class MonitoringConfigTest(unittest.TestCase):
    def test_progress_states_uses_http_and_unbuffered_python(self) -> None:
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")
        dashboard = Path("monitoring/grafana/dashboards/processing-progress.json").read_text(encoding="utf-8")
        readme = Path("monitoring/grafana/README.md").read_text(encoding="utf-8")

        self.assertIn("PYTHONUNBUFFERED: 1", compose)
        self.assertIn("http://progress-states:8000/processing-states.json", dashboard)
        self.assertNotIn("https://progress-states:8000/processing-states.json", dashboard)
        self.assertIn("http://localhost:8000/processing-states.json", readme)

    def test_dashboard_dataset_variable_includes_configured_datasets(self) -> None:
        dashboard = json.loads(Path("monitoring/grafana/dashboards/processing-progress.json").read_text(encoding="utf-8"))
        variables = {variable["name"]: variable for variable in dashboard["templating"]["list"]}
        dataset = variables["dataset"]

        self.assertEqual(dataset["type"], "query")
        self.assertEqual(dataset["datasource"]["type"], "yesoreyeram-infinity-datasource")
        self.assertEqual(dataset["query"]["url"], "http://progress-states:8000/datasets.json")
        self.assertEqual(dataset["query"]["root_selector"], "$")
        self.assertNotIn("custom", dataset["query"])


if __name__ == "__main__":
    unittest.main()
