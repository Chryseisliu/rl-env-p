from collections import deque
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count

import numpy as np
import xxhash


@dataclass
class Config:
    max_num_batched_tokens: int = 16
    max_num_seqs: int = 4
    max_model_len: int = 128
    eos: int = -1
    kvcache_block_size: int = 4
    num_kvcache_blocks: int = 32


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 4
    ignore_eos: bool = False


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 4
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, index: int) -> list[int]:
        start = index * self.block_size
        stop = (index + 1) * self.block_size
        return self.token_ids[start:stop]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, block_hash: int, token_ids: list[int]) -> None:
        self.hash = block_hash
        self.token_ids = token_ids

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        hasher = xxhash.xxh64()
        if prefix != -1:
            hasher.update(prefix.to_bytes(8, "little"))
        hasher.update(np.array(token_ids, dtype=np.int64).tobytes())
        return hasher.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        cached_tokens = 0
        prefix_hash = -1
        cache_miss = False

        for index in range(seq.num_blocks):
            token_ids = seq.block(index)
            prefix_hash = self.compute_hash(token_ids, prefix_hash) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(prefix_hash, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                break
            cached_tokens += self.block_size

        return cached_tokens

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        prefix_hash = -1
        cache_miss = False

        for index in range(seq.num_blocks):
            token_ids = seq.block(index)
            prefix_hash = self.compute_hash(token_ids, prefix_hash) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(prefix_hash, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            if prefix_hash != -1:
                block.update(prefix_hash, token_ids)
                self.hash_to_block_id[prefix_hash] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        needs_new_block = len(seq) % self.block_size == 1
        return len(self.free_block_ids) >= int(needs_new_block)

    def may_append(self, seq: Sequence) -> None:
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            block_hash = self.compute_hash(token_ids, prefix)
            last_block.update(block_hash, token_ids)
            self.hash_to_block_id[block_hash] = last_block.block_id


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        Sequence.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled: list[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            cached_tokens = self.block_manager.get_num_cached_tokens(seq)
            uncached_tokens = len(seq) - cached_tokens
            if (
                num_batched_tokens + uncached_tokens > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)
            ):
                break

            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += uncached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)

        if scheduled:
            return scheduled, True

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled.append(seq)

        if not scheduled:
            raise RuntimeError("scheduler could not make progress")
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


class ModelRunner:
    def __init__(self, config: Config):
        self.config = config

    def call(self, method_name: str, *args):
        return getattr(self, method_name)(*args)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        step_bias = 3 if is_prefill else 1
        return [((seq.last_token + step_bias) % 997) + 1 for seq in seqs]


class LLMEngine:
    def __init__(self, config: Config):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config)

    def add_request(self, prompt: list[int], sampling_params: SamplingParams) -> None:
        self.scheduler.add(Sequence(prompt, sampling_params))

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
