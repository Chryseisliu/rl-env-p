from nanovllm import Config, LLMEngine, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def test_block_manager_reuses_full_prefix_blocks() -> None:
    Sequence.block_size = 4
    manager = BlockManager(num_blocks=8, block_size=4)

    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=1))
    manager.allocate(seq)
    manager.deallocate(seq)

    reused = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9], SamplingParams(max_tokens=1))
    manager.allocate(reused)

    assert reused.num_cached_tokens == 8
    assert len(reused.block_table) == 3


def test_scheduler_makes_progress_without_prefix_caching_pressure() -> None:
    scheduler = Scheduler(Config(max_num_batched_tokens=8, max_num_seqs=2, kvcache_block_size=4, num_kvcache_blocks=8))
    seq = Sequence([10, 11, 12, 13], SamplingParams(max_tokens=1))
    scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert [item.seq_id for item in scheduled] == [seq.seq_id]
    assert seq.num_cached_tokens == 0



def test_scheduler_prefill_budget_should_count_uncached_tokens() -> None:
    config = Config(max_num_batched_tokens=2, max_num_seqs=2, kvcache_block_size=4, num_kvcache_blocks=8)
    scheduler = Scheduler(config)

    seed = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=1))
    scheduler.block_manager.allocate(seed)
    scheduler.block_manager.deallocate(seed)

    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 99, 100], SamplingParams(max_tokens=1))
    scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert [item.seq_id for item in scheduled] == [seq.seq_id]
    assert scheduled[0].num_cached_tokens == 8


def test_engine_generate_returns_one_completion_per_request() -> None:
    engine = LLMEngine(Config(max_num_batched_tokens=12, max_num_seqs=2, kvcache_block_size=4, num_kvcache_blocks=8))
    outputs = engine.generate(
        prompts=[[1, 2, 3, 4], [5, 6, 7, 8]],
        sampling_params=SamplingParams(max_tokens=2),
    )

    assert len(outputs) == 2
    assert all(len(item["token_ids"]) == 2 for item in outputs)
