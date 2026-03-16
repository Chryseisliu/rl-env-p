from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, config: Config):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config)

    def add_request(self, prompt: list[int], sampling_params: SamplingParams) -> None:
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) - seq.num_cached_tokens for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, params in zip(prompts, sampling_params):
            self.add_request(prompt, params)

        outputs: dict[int, list[int]] = {}
        while not self.is_finished():
            finished, _ = self.step()
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids

        return [{"token_ids": outputs[seq_id]} for seq_id in sorted(outputs.keys())]
