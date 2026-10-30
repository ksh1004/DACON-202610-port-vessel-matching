"""Score catalog spellings under the existing transliteration LLM.

The prompt KV cache is shared only among candidates for one query. A fresh cache
is created on every call; no candidate or query can affect another query's score.
"""
from __future__ import annotations

import numpy as np
import torch

from src.llm_translit import Transliterator


class ClosedNameScorer:
    def __init__(self, model_dir):
        self.translit = Transliterator(model_dir)
        if self.translit.seq2seq:
            raise ValueError("Closed-name scoring requires a causal model")

    @torch.inference_mode()
    def score(self, query: str, names: list[str]) -> np.ndarray:
        """Return [mean token log probability, total log probability] per name."""
        if not names:
            return np.empty((0, 2), dtype=np.float32)
        t = self.translit
        enc = t.tokenizer(t._prompt(query), return_tensors="pt",
                          add_special_tokens=t.add_special_tokens).to(t.device)
        enc.pop("token_type_ids", None)
        prefix = t.model(**enc, use_cache=True, logits_to_keep=1)
        first = prefix.logits[0, -1].float().log_softmax(-1)
        cache = prefix.past_key_values
        cache.batch_repeat_interleave(len(names))
        targets = [t.tokenizer.encode(n, add_special_tokens=False) + [t.tokenizer.eos_token_id] for n in names]
        lengths = [len(ids) for ids in targets]
        width = max(max(lengths) - 1, 1)
        suffix = torch.full((len(names), width), t.tokenizer.pad_token_id, dtype=torch.long, device=t.device)
        suffix_mask = torch.zeros_like(suffix)
        for i, ids in enumerate(targets):
            if len(ids) > 1:
                suffix[i, :len(ids) - 1] = torch.tensor(ids[:-1], device=t.device)
                suffix_mask[i, :len(ids) - 1] = 1
        mask = torch.cat([torch.ones((len(names), enc.input_ids.shape[1]), dtype=torch.long, device=t.device),
                          suffix_mask], dim=1)
        out = t.model(input_ids=suffix, attention_mask=mask, past_key_values=cache, use_cache=False)
        logits = out.logits.float().log_softmax(-1)
        values = []
        for i, ids in enumerate(targets):
            total = first[ids[0]]
            if len(ids) > 1:
                total = total + logits[i, torch.arange(len(ids) - 1, device=t.device),
                                       torch.tensor(ids[1:], device=t.device)].sum()
            value = float(total.item())
            values.append((value / len(ids), value))
        return np.array(values, dtype=np.float32)
