"""Query-local byte-level probabilities of observed Latin name fragments."""
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.common import normalize_name

MODEL_DIR = Path("D:/KSH/hf_models/google__byt5-small")
PREFIX = "latin: "


def fragments(name):
    whole = normalize_name(name)
    words = whole.split()
    if not words:
        raise ValueError("Empty candidate name")
    distinct = {" ".join(words[i:j]) for i in range(len(words)) for j in range(i+1, len(words)+1)}
    other = sorted(distinct - {whole})
    if not other:
        return [(whole, 1.)]
    return [(whole, .5)] + [(s, .5 / len(other)) for s in other]


def encode(tokenizer, pair):
    query, target = pair
    return (tokenizer.encode(PREFIX + normalize_name(query), add_special_tokens=True),
            tokenizer.encode(normalize_name(target), add_special_tokens=True))


def bucket(length, minimum, step):
    return max(minimum, step * math.ceil(length / step))


def tensors(encoded, device, source_length=None, target_length=None):
    source_length = source_length or bucket(max(len(x) for x, y in encoded), 128, 64)
    target_length = target_length or bucket(max(len(y) for x, y in encoded), 96, 32)
    ids = torch.zeros((len(encoded), source_length), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full((len(encoded), target_length), -100, dtype=torch.long, device=device)
    decoder_mask = torch.zeros_like(labels)
    for i, (x, y) in enumerate(encoded):
        if len(x) > source_length or len(y) > target_length:
            raise ValueError("No byte-name truncation is allowed")
        ids[i, :len(x)] = torch.tensor(x, device=device)
        mask[i, :len(x)] = 1
        labels[i, :len(y)] = torch.tensor(y, device=device)
        decoder_mask[i, :len(y)] = 1
    decoder = torch.zeros_like(labels)
    decoder[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, 0)
    return {"input_ids": ids, "attention_mask": mask, "decoder_input_ids": decoder,
            "decoder_attention_mask": decoder_mask}, labels


def sequence_logp(logits, labels):
    valid = labels != -100
    safe = labels.masked_fill(~valid, 0)
    values = torch.log_softmax(logits.float(), -1).gather(-1, safe[..., None]).squeeze(-1)
    return values.masked_fill(~valid, 0), valid


def probability(scores):
    scores = np.asarray(scores, dtype=np.float64)
    values = np.exp(scores - scores.max())
    return values / np.sort(values).sum(dtype=np.float64)


def normalise(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.sort(values).sum(dtype=np.float64)


class ByteNameScorer:
    def __init__(self, model_dir=MODEL_DIR, device="cuda", batch_size=16):
        self.model_dir, self.device, self.batch_size = Path(model_dir), device, int(batch_size)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager").to(device).eval()
        self.model.config.use_cache = False

    @torch.inference_mode()
    def score_pairs(self, pairs):
        output = np.empty(len(pairs), dtype=np.float64)
        batches = defaultdict(list)
        for i, pair in enumerate(pairs):
            x, y = encode(self.tokenizer, pair)
            batches[(bucket(len(x), 128, 64), bucket(len(y), 96, 32))].append((i, x, y))
        for (sx, sy), items in sorted(batches.items()):
            for start in range(0, len(items), self.batch_size):
                part = items[start:start+self.batch_size]
                encoded = [(x, y) for i, x, y in part]
                encoded += [encoded[-1]] * (self.batch_size-len(encoded))
                inputs, labels = tensors(encoded, self.device, sx, sy)
                values, valid = sequence_logp(self.model(**inputs, use_cache=False).logits, labels)
                values = values.cpu().numpy()
                for j, (i, x, y) in enumerate(part):
                    output[i] = values[j, :len(y)].sum(dtype=np.float64)
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite byte-name likelihood")
        return output

    def score(self, query, candidate_names):
        parts = [fragments(name) for name in candidate_names]
        names = sorted({s for group in parts for s, p in group})
        values = dict(zip(names, self.score_pairs([(query, s) for s in names])))
        return np.array([np.logaddexp.reduce([values[s] + math.log(p) for s, p in group])
                         for group in parts], dtype=np.float64)
