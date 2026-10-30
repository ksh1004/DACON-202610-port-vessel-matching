"""Training-length-normalised compatibility, distinct from sequence probability."""
import math

import numpy as np

from src.byte_name_model import ByteNameScorer, fragments
from src.common import normalize_name


def target_length(name):
    return len(normalize_name(name).encode("utf-8")) + 1


def length_scale(examples):
    return float(np.median([target_length(r["target"]) for r in examples]))


def aggregate(query, names, cache, scale):
    return np.array([np.logaddexp.reduce([cache[(query,s)] * scale/target_length(s) + math.log(p)
                                         for s,p in fragments(name)]) for name in names],dtype=np.float64)


class LengthByteNameScorer(ByteNameScorer):
    def __init__(self, model_dir, scale, **kwargs):
        super().__init__(model_dir, **kwargs)
        self.scale = float(scale)

    def score(self, query, candidate_names):
        spans = sorted({s for name in candidate_names for s,p in fragments(name)})
        cache = dict(zip([(query,s) for s in spans],self.score_pairs([(query,s) for s in spans])))
        return aggregate(query,candidate_names,cache,self.scale)
