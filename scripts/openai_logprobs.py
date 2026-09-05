from __future__ import annotations

from typing import Any
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
MINISTRAL_Q4_FILES = {
    "unsloth/Ministral-3-3B-Instruct-2512-GGUF": "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    "unsloth/Ministral-3-3B-Reasoning-2512-GGUF": "Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf",
}
MINISTRAL_TOKENIZER_IDS = {
    model_id: model_id.replace("unsloth/", "mistralai/").replace("-GGUF", "")
    for model_id in MINISTRAL_Q4_FILES
}


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _normalize_top_logprobs(
    tokenizer: Any,
    log_probs: torch.Tensor,
    top_n: int,
) -> list[dict[str, Any]]:
    top_k = min(top_n, log_probs.shape[-1])
    top_values, top_indices = torch.topk(log_probs, k=top_k)

    entries = [
        {"token": _decode_token(tokenizer, idx), "token_id": int(idx), "logprob": float(value)}
        for value, idx in zip(top_values.tolist(), top_indices.tolist())
    ]

    if len(entries) < top_n:
        entries.extend({"token": None, "token_id": None, "logprob": None} for _ in range(top_n - len(entries)))

    return entries


def load_local_model(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any, torch.device]:
    local_files_only = (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in TRUE_ENV_VALUES
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in TRUE_ENV_VALUES
    )
    load_kwargs = {"local_files_only": True} if local_files_only else {}

    tokenizer_kwargs = dict(load_kwargs)
    if model_id in MINISTRAL_Q4_FILES:
        tokenizer_kwargs["use_fast"] = False
    tokenizer_id = MINISTRAL_TOKENIZER_IDS.get(model_id, model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
    model_kwargs = dict(load_kwargs)
    if gguf_file := MINISTRAL_Q4_FILES.get(model_id):
        model_kwargs["gguf_file"] = gguf_file
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def score_texts_with_model(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    top_n: int = 20,
    inference_batch_size: int = 4,
    progress_desc: str | None = None,
) -> list[dict[str, Any]]:
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be at least 1")

    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if start_token_id is None:
        raise RuntimeError("Tokenizer does not define a BOS/EOS token needed to score prompt tokens.")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = start_token_id

    results: list[dict[str, Any]] = []

    for start in tqdm(
        range(0, len(texts), inference_batch_size),
        desc=progress_desc,
        unit="batch",
        disable=progress_desc is None,
    ):
        batch_texts = texts[start : start + inference_batch_size]
        batch_token_ids = [
            tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            for text in batch_texts
        ]
        non_empty = [(text, token_ids) for text, token_ids in zip(batch_texts, batch_token_ids) if token_ids.numel() > 0]

        batch_results: list[dict[str, Any] | None] = [None] * len(batch_texts)
        for index, token_ids in enumerate(batch_token_ids):
            if token_ids.numel() == 0:
                batch_results[index] = {"text": batch_texts[index], "tokens": []}

        if non_empty:
            full_sequences = [torch.cat([torch.tensor([start_token_id], dtype=torch.long), token_ids]) for _, token_ids in non_empty]
            max_length = max(sequence.numel() for sequence in full_sequences)
            input_ids = torch.full((len(full_sequences), max_length), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(full_sequences), max_length), dtype=torch.long)

            for index, sequence in enumerate(full_sequences):
                input_ids[index, : sequence.numel()] = sequence
                attention_mask[index, : sequence.numel()] = 1

            with torch.inference_mode():
                logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits.cpu()

            non_empty_index = 0
            for batch_index, token_ids in enumerate(batch_token_ids):
                if token_ids.numel() == 0:
                    continue

                log_probs = torch.log_softmax(logits[non_empty_index, : token_ids.numel()], dim=-1)
                non_empty_index += 1

                token_records: list[dict[str, Any]] = []
                for position, token_id in enumerate(token_ids.tolist()):
                    token_logprob = log_probs[position, token_id].item()
                    rank = int((log_probs[position] > token_logprob).sum().item()) + 1
                    token_records.append(
                        {
                            "token": _decode_token(tokenizer, token_id),
                            "token_id": token_id,
                            "logprob": token_logprob,
                            "rank": rank,
                            "top_logprobs": _normalize_top_logprobs(tokenizer, log_probs[position], top_n),
                        }
                    )

                batch_results[batch_index] = {"text": batch_texts[batch_index], "tokens": token_records}

        results.extend(result for result in batch_results if result is not None)

    return results


def process_text_batch(
    texts: list[str],
    model_id: str = DEFAULT_MODEL_ID,
    top_n: int = 20,
    inference_batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Return exact per-token logprobs for a batch of texts using a local causal LM."""
    tokenizer, model, device = load_local_model(model_id)
    return score_texts_with_model(texts, tokenizer, model, device, top_n, inference_batch_size=inference_batch_size)


if __name__ == "__main__":
    sample = process_text_batch(["This is a test sentence."])
    print(sample)
