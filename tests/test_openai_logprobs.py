from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.openai_logprobs import load_local_model


class LoadLocalModelTest(unittest.TestCase):
    def test_uses_local_files_only_when_hf_hub_offline_is_enabled(self) -> None:
        class DummyModel:
            def eval(self) -> None:
                pass

            def to(self, device: object) -> None:
                pass

        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
            with patch("scripts.openai_logprobs.AutoTokenizer.from_pretrained") as tokenizer_loader:
                with patch("scripts.openai_logprobs.AutoModelForCausalLM.from_pretrained") as model_loader:
                    model_loader.return_value = DummyModel()

                    load_local_model("facebook/opt-350m")

        tokenizer_loader.assert_called_once_with("facebook/opt-350m", local_files_only=True)
        model_loader.assert_called_once_with("facebook/opt-350m", local_files_only=True)

    def test_selects_q4_gguf_file_for_ministral_repository(self) -> None:
        class DummyModel:
            def eval(self) -> None:
                pass

            def to(self, device: object) -> None:
                pass

        model_id = "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
            with patch("scripts.openai_logprobs.AutoTokenizer.from_pretrained") as tokenizer_loader:
                with patch("scripts.openai_logprobs.AutoModelForCausalLM.from_pretrained") as model_loader:
                    model_loader.return_value = DummyModel()

                    load_local_model(model_id)

        tokenizer_loader.assert_called_once_with(model_id, local_files_only=True)
        model_loader.assert_called_once_with(
            model_id,
            local_files_only=True,
            gguf="Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        )


if __name__ == "__main__":
    unittest.main()
