from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class ScriptEnvLoadingTest(unittest.TestCase):
    def test_experiment_config_loads_text_detection_data_from_dotenv(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(repo_root / "scripts")

        dotenv_data = Path("D:/tmp/text-detection-dotenv-data")

        with patch.dict(os.environ, {}, clear=True):
            old_path = sys.path.copy()
            sys.modules.pop("experiment_config", None)
            try:
                sys.path.insert(0, scripts_path)
                with patch("dotenv.load_dotenv") as load_dotenv:
                    def set_env(*_: object, **__: object) -> None:
                        os.environ["TEXT_DETECTION_DATA"] = str(dotenv_data)

                    load_dotenv.side_effect = set_env
                    experiment_config = importlib.import_module("experiment_config")
            finally:
                sys.path[:] = old_path
                sys.modules.pop("experiment_config", None)

        load_dotenv.assert_called_once()
        self.assertEqual(experiment_config.DATA_ROOT, dotenv_data)


if __name__ == "__main__":
    unittest.main()
