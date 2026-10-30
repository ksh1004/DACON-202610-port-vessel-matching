"""Frozen Korean-to-Latin candidate likelihood using the existing public LM."""
import hashlib
from pathlib import Path

from src.common import normalize_name
from src.llm_translit import FEW_SHOT, SYSTEM_PROMPT
from src.reverse_name_likelihood import ReverseNameScorer

PROMPT = SYSTEM_PROMPT + " Use UPPERCASE Latin letters. Render only the name words present in the input; do not invent missing words."


class ForwardNameScorer(ReverseNameScorer):
    def __init__(self, **kwargs):
        super().__init__(batch_size=32, pad_length=512, **kwargs)
        # End the candidate with the assistant-turn terminator used by this chat
        # template, which need not be the tokenizer's end-of-text token.
        messages = self.messages("TEST")
        prefix = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        complete = self.tokenizer.apply_chat_template(messages+[{"role":"assistant","content":"X"}], tokenize=True,
                                                     add_generation_prompt=False, enable_thinking=False)
        target = self.tokenizer.encode("X", add_special_tokens=False)
        assert complete[:len(prefix)] == prefix
        assert complete[len(prefix):len(prefix)+len(target)] == target
        terminator = int(complete[len(prefix)+len(target)])
        assert terminator in self.tokenizer.all_special_ids
        self.eos_token_id = terminator

    def messages(self, query):
        messages = [{"role":"system","content":PROMPT}]
        for ko,en in FEW_SHOT:
            messages += [{"role":"user","content":ko},{"role":"assistant","content":normalize_name(en)}]
        return messages+[{"role":"user","content":query}]

    def prefix_ids(self, query):
        return self.tokenizer.apply_chat_template(self.messages(query), tokenize=True,
                                                 add_generation_prompt=True, enable_thinking=False)

    def score_pairs(self, pairs):
        # The shared fixed-shape implementation takes (target, conditioning).
        values = self.score_many_pairs([(normalize_name(span),query) for query,span in pairs])
        return values[:,0]+values[:,2]

    def protocol(self):
        p=super().protocol()
        p.update(direction="P(uppercase Latin fragment + assistant turn end | observed Korean)",prompt=PROMPT,
                 few_shot_examples=len(FEW_SHOT),examples=[(q,normalize_name(v)) for q,v in FEW_SHOT],
                 target_terminator_id=self.eos_token_id,
                 own_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return p
