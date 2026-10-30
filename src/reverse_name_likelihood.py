"""Teacher-forced P(observed Korean name | official Latin candidate name).

Only name strings enter the model. One call processes one query's candidates;
no other input rows are used. Candidate order is canonicalized, and both the
microbatch and padded sequence shape are fixed. Across different batch sizes,
BF16 values can still differ: keep these settings fixed in a release and do not
mix pair caches obtained under different batching protocols.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path("D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505")
SYSTEM_PROMPT = (
    "Transliterate the entire Latin-alphabet ship name phonetically into Korean Hangul. "
    "Render every name word and preserve all digits. Do not translate the meaning "
    "or add information. Output only the Korean transliteration."
)


class ReverseNameScorer:
    """Offline name likelihood with an explicit, reproducible batching recipe."""

    def __init__(self, model_dir: str | Path = MODEL_DIR, device: str = "cuda",
                 batch_size: int = 32, pad_length: int = 128,
                 dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "sdpa"):
        if batch_size < 1 or pad_length < 8:
            raise ValueError("batch_size must be positive and pad_length at least 8")
        self.model_dir = Path(model_dir)
        self.device = device
        self.batch_size = int(batch_size)
        self.pad_length = int(pad_length)
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_target_tokens = 32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("A tokenizer EOS token is required for the separate EOS score")
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, local_files_only=True, dtype=dtype,
            attn_implementation=attn_implementation).to(device).eval()
        self.model.config.use_cache = False
        if self.pad_length > int(self.model.config.max_position_embeddings):
            raise ValueError("pad_length exceeds the model context size")

    def protocol(self) -> dict:
        return {
            "model_dir": str(self.model_dir), "prompt": SYSTEM_PROMPT,
            "few_shot_examples": 0, "input_fields": ["observed_name", "candidate_name"],
            "batch_size": self.batch_size, "pad_length": self.pad_length,
            "padding_side": "left", "padded_last_batch": "repeat last candidate",
            "candidate_order": "sorted unique raw name strings",
            "position_ids": "attention-mask cumulative positions, padded positions zero",
            "dtype": str(self.dtype), "attention": self.attn_implementation,
            "target_tokenization": "separate target encode, add_special_tokens=False",
            "fixed_logits_to_keep": self.max_target_tokens + 1,
            "maximum_target_tokens": self.max_target_tokens,
            "score": "observed target tokens only; EOS reported separately",
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_config_sha256": hashlib.sha256((self.model_dir / "config.json").read_bytes()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256((self.model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        }

    def prefix_ids(self, candidate_name: str) -> list[int]:
        if not isinstance(candidate_name, str) or not candidate_name.strip():
            raise ValueError("Candidate name must be a nonempty string")
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": candidate_name}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def target_ids(self, observed_name: str) -> list[int]:
        if not isinstance(observed_name, str) or not observed_name.strip():
            raise ValueError("Observed name must be a nonempty string")
        result = self.tokenizer.encode(observed_name, add_special_tokens=False)
        if not result:
            raise ValueError("Observed name has no tokens")
        if len(result) > self.max_target_tokens:
            raise ValueError("Target exceeds 32 tokens; no truncation performed")
        return result

    @torch.inference_mode()
    def score_many_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return [sum, mean, EOS, token count] for each pair, in input order.

        All model tensor shapes, including the vocabulary projection, are fixed.
        Pair caching is enabled only after the independent invariance check.
        """
        if not pairs:
            return np.empty((0, 4), dtype=np.float64)
        output = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start+self.batch_size]
            targets = [self.target_ids(q) for q, _ in chunk]
            sequences = [self.prefix_ids(c)+t for (_, c), t in zip(chunk, targets)]
            if max(map(len, sequences)) > self.pad_length:
                raise ValueError("Input exceeds fixed pad_length; no truncation performed")
            actual = len(sequences)
            sequences += [sequences[-1]]*(self.batch_size-actual)
            ids = torch.full((self.batch_size, self.pad_length), self.tokenizer.pad_token_id,
                             dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(sequences):
                ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
                mask[i, -len(seq):] = 1
            logits = self.model(input_ids=ids, attention_mask=mask,
                                position_ids=(mask.cumsum(-1)-1).clamp_min(0),
                                use_cache=False, logits_to_keep=self.max_target_tokens+1).logits.float()
            for i, target in enumerate(targets):
                tail = logits[i, -len(target)-1:]
                labels = torch.tensor(target+[self.eos_token_id], device=self.device)
                lp = tail.gather(-1, labels[:, None]).squeeze(-1)-torch.logsumexp(tail, -1)
                values = lp.cpu().numpy()
                total = float(values[:-1].sum(dtype=np.float64))
                output.append((total, total/len(target), float(values[-1]), len(target)))
        result = np.asarray(output, dtype=np.float64)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite reverse-name log likelihood")
        return result

    def encode(self, observed_name: str, candidate_names: Sequence[str],
               repeat_to_batch: bool = True) -> tuple[dict[str, torch.Tensor], list[int], int]:
        """Expose the exact tokens for independent score verification."""
        target = self.target_ids(observed_name)
        sequences = [self.prefix_ids(name)+target for name in candidate_names]
        if not sequences or len(sequences) > self.batch_size:
            raise ValueError("encode requires between 1 and batch_size candidates")
        maximum = max(map(len, sequences))
        if maximum > self.pad_length:
            raise ValueError(f"Input has {maximum} tokens, above fixed pad_length={self.pad_length}; no truncation performed")
        actual = len(sequences)
        if repeat_to_batch:
            sequences += [sequences[-1]]*(self.batch_size-actual)
        ids = torch.full((len(sequences), self.pad_length), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, -len(seq):] = torch.tensor(seq, dtype=torch.long, device=self.device)
            mask[i, -len(seq):] = 1
        positions = (mask.cumsum(-1)-1).clamp_min(0)
        return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}, target, actual

    @torch.inference_mode()
    def score_details(self, observed_name: str, candidate_names: Sequence[str]) -> dict:
        candidates = list(candidate_names)
        target = self.target_ids(observed_name)
        if not candidates:
            return {"sum_logp": np.empty(0), "mean_logp": np.empty(0),
                    "eos_logp": np.empty(0), "sum_with_eos": np.empty(0),
                    "mean_with_eos": np.empty(0), "target_tokens": len(target)}
        # Sorting and deduplication make candidate permutation/chunk composition
        # deterministic for a given query's candidate set.
        unique = sorted(set(candidates))
        array4 = self.score_many_pairs([(observed_name, name) for name in unique])
        results = {name: (v[0], v[1], v[2], v[0]+v[2], (v[0]+v[2])/(v[3]+1))
                   for name, v in zip(unique, array4)}
        array = np.asarray([results[name] for name in candidates], dtype=np.float64)
        return {"sum_logp": array[:, 0], "mean_logp": array[:, 1], "eos_logp": array[:, 2],
                "sum_with_eos": array[:, 3], "mean_with_eos": array[:, 4], "target_tokens": len(target)}
    def score(self, observed_name: str, candidate_names: Sequence[str]) -> np.ndarray:
        return self.score_details(observed_name, candidate_names)["mean_logp"]

