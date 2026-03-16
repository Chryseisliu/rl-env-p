from nanovllm import Config, SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def main() -> None:
    scheduler = Scheduler(
        Config(
            max_num_batched_tokens=2,
            max_num_seqs=2,
            kvcache_block_size=4,
            num_kvcache_blocks=8,
        )
    )

    seed = Sequence([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_tokens=1))
    scheduler.block_manager.allocate(seed)
    scheduler.block_manager.deallocate(seed)

    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 99, 100], SamplingParams(max_tokens=1))
    scheduler.add(seq)

    try:
        scheduled, is_prefill = scheduler.schedule()
        print("scheduled ids:", [item.seq_id for item in scheduled])
        print("is_prefill:", is_prefill)
        print("cached tokens after allocation:", scheduled[0].num_cached_tokens)
    except Exception as exc:
        print("scheduler failed:", repr(exc))


if __name__ == "__main__":
    main()
