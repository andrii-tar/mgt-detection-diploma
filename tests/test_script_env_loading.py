from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class ScriptEnvLoadingTest(unittest.TestCase):
    def test_resolve_data_root_loads_dotenv(self) -> None:
        from scripts.data_root import resolve_data_root

        dotenv_data = Path("D:/tmp/text-detection-dotenv-data")
        with patch.dict(os.environ, {}, clear=True), patch("scripts.data_root.load_dotenv") as load_dotenv:
            def set_env(*_: object, **__: object) -> None:
                os.environ["TEXT_DETECTION_DATA"] = str(dotenv_data)

            load_dotenv.side_effect = set_env
            self.assertEqual(resolve_data_root(), dotenv_data)

        load_dotenv.assert_called_once()

    def test_resolve_data_root_raises_canonical_error_when_unset(self) -> None:
        from scripts.data_root import resolve_data_root

        with patch.dict(os.environ, {}, clear=True), patch("scripts.data_root.load_dotenv"):
            with self.assertRaisesRegex(RuntimeError, "^TEXT_DETECTION_DATA is not set$"):
                resolve_data_root()

    def test_experiment_config_import_does_not_require_data_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(repo_root / "scripts")

        with patch.dict(os.environ, {}, clear=True):
            old_path = sys.path.copy()
            sys.modules.pop("experiment_config", None)
            try:
                sys.path.insert(0, scripts_path)
                experiment_config = importlib.import_module("experiment_config")
            finally:
                sys.path[:] = old_path
                sys.modules.pop("experiment_config", None)

        self.assertIn("kaggle_train", experiment_config.DATASETS)


if __name__ == "__main__":
    unittest.main()
