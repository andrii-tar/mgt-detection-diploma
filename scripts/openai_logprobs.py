from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


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
        {"token": _decode_token(tokenizer, idx), "logprob": float(value)}
        for value, idx in zip(top_values.tolist(), top_indices.tolist())
    ]

    if len(entries) < top_n:
        entries.extend({"token": None, "logprob": None} for _ in range(top_n - len(entries)))

    return entries


def process_text_batch(texts: list[str], model_id: str = DEFAULT_MODEL_ID, top_n: int = 20) -> list[dict[str, Any]]:
    """Return exact per-token logprobs for a batch of texts using a local causal LM."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if start_token_id is None:
        raise RuntimeError("Tokenizer does not define a BOS/EOS token needed to score prompt tokens.")

    results: list[dict[str, Any]] = []

    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]

        if token_ids.numel() == 0:
            results.append({"text": text, "tokens": []})
            continue

        prefix = torch.tensor([start_token_id], dtype=torch.long)
        full_ids = torch.cat([prefix, token_ids]).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(input_ids=full_ids).logits[0]

        log_probs = torch.log_softmax(logits[:-1], dim=-1)

        token_records: list[dict[str, Any]] = []
        for position, token_id in enumerate(token_ids.tolist()):
            token_logprob = log_probs[position, token_id].item()
            rank = int((log_probs[position] > token_logprob).sum().item()) + 1
            token_records.append(
                {
                    "token": _decode_token(tokenizer, token_id),
                    "logprob": token_logprob,
                    "rank": rank,
                    "top_logprobs": _normalize_top_logprobs(tokenizer, log_probs[position], top_n),
                }
            )

        results.append({"text": text, "tokens": token_records})

    return results


if __name__ == "__main__":
    sample = process_text_batch(["This is a test sentence."])
    print(sample)
