from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence


class ModelRunner:
    def __init__(self, config: Config):
        self.config = config

    def call(self, method_name: str, *args):
        method = getattr(self, method_name)
        return method(*args)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return [self._next_token(seq, is_prefill) for seq in seqs]

    def _next_token(self, seq: Sequence, is_prefill: bool) -> int:
        step_bias = 1 if not is_prefill else 3
        return ((seq.last_token + step_bias) % 997) + 1
