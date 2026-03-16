import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from reference_impl import Config as RefConfig
from reference_impl import LLMEngine as RefLLMEngine
from reference_impl import SamplingParams as RefSamplingParams
from reference_impl import Scheduler as RefScheduler
from reference_impl import Sequence as RefSequence


@dataclass
class CaseResult:
    name: str
    weight: int
    passed: bool
    detail: str


def _load_candidate_modules(candidate_repo: Path):
    candidate_path = str(candidate_repo)
    if candidate_path not in sys.path:
        sys.path.insert(0, candidate_path)

    for module_name in list(sys.modules):
        if module_name == "nanovllm" or module_name.startswith("nanovllm."):
            sys.modules.pop(module_name, None)

    package = importlib.import_module("nanovllm")
    scheduler_module = importlib.import_module("nanovllm.engine.scheduler")
    sequence_module = importlib.import_module("nanovllm.engine.sequence")
    return package, scheduler_module, sequence_module


def _seed_prefix_cache(scheduler, sequence_cls, sampling_params_cls) -> None:
    seed = sequence_cls([1, 2, 3, 4, 5, 6, 7, 8], sampling_params_cls(max_tokens=1))
    scheduler.block_manager.allocate(seed)
    scheduler.block_manager.deallocate(seed)


def _run_shared_prefix_single(candidate_package, scheduler_module, sequence_module) -> tuple[bool, str]:
    config = candidate_package.Config(
        max_num_batched_tokens=2,
        max_num_seqs=2,
        kvcache_block_size=4,
        num_kvcache_blocks=8,
    )
    candidate_scheduler = scheduler_module.Scheduler(config)
    _seed_prefix_cache(candidate_scheduler, sequence_module.Sequence, candidate_package.SamplingParams)

    reference_scheduler = RefScheduler(RefConfig(**config.__dict__))
    _seed_prefix_cache(reference_scheduler, RefSequence, RefSamplingParams)

    candidate_seq = sequence_module.Sequence([1, 2, 3, 4, 5, 6, 7, 8, 99, 100], candidate_package.SamplingParams(max_tokens=1))
    reference_seq = RefSequence([1, 2, 3, 4, 5, 6, 7, 8, 99, 100], RefSamplingParams(max_tokens=1))
    candidate_scheduler.add(candidate_seq)
    reference_scheduler.add(reference_seq)

    try:
        candidate_scheduled, candidate_is_prefill = candidate_scheduler.schedule()
        candidate_detail = f"scheduled={len(candidate_scheduled)}, cached={candidate_scheduled[0].num_cached_tokens if candidate_scheduled else 0}, is_prefill={candidate_is_prefill}"
    except Exception as exc:
        candidate_scheduled = None
        candidate_is_prefill = None
        candidate_detail = f"exception={type(exc).__name__}: {exc}"

    reference_scheduled, reference_is_prefill = reference_scheduler.schedule()
    reference_detail = f"scheduled={len(reference_scheduled)}, cached={reference_scheduled[0].num_cached_tokens}, is_prefill={reference_is_prefill}"

    passed = (
        candidate_scheduled is not None
        and len(candidate_scheduled) == len(reference_scheduled) == 1
        and candidate_scheduled[0].num_cached_tokens == reference_scheduled[0].num_cached_tokens == 8
        and candidate_is_prefill == reference_is_prefill is True
    )
    return passed, f"candidate({candidate_detail}) reference({reference_detail})"


def _run_shared_prefix_batch(candidate_package, scheduler_module, sequence_module) -> tuple[bool, str]:
    config = candidate_package.Config(
        max_num_batched_tokens=4,
        max_num_seqs=2,
        kvcache_block_size=4,
        num_kvcache_blocks=12,
    )
    candidate_scheduler = scheduler_module.Scheduler(config)
    _seed_prefix_cache(candidate_scheduler, sequence_module.Sequence, candidate_package.SamplingParams)

    reference_scheduler = RefScheduler(RefConfig(**config.__dict__))
    _seed_prefix_cache(reference_scheduler, RefSequence, RefSamplingParams)

    prompts = [
        [1, 2, 3, 4, 5, 6, 7, 8, 41, 42],
        [1, 2, 3, 4, 5, 6, 7, 8, 51, 52],
    ]
    for prompt in prompts:
        candidate_scheduler.add(sequence_module.Sequence(prompt, candidate_package.SamplingParams(max_tokens=1)))
        reference_scheduler.add(RefSequence(prompt, RefSamplingParams(max_tokens=1)))

    try:
        candidate_scheduled, _ = candidate_scheduler.schedule()
        candidate_detail = f"scheduled={len(candidate_scheduled)}"
    except Exception as exc:
        candidate_scheduled = None
        candidate_detail = f"exception={type(exc).__name__}: {exc}"

    reference_scheduled, _ = reference_scheduler.schedule()
    reference_detail = f"scheduled={len(reference_scheduled)}"
    passed = candidate_scheduled is not None and len(candidate_scheduled) == len(reference_scheduled) == 2
    return passed, f"candidate({candidate_detail}) reference({reference_detail})"


def _run_end_to_end(candidate_package, sequence_module) -> tuple[bool, str]:
    config = candidate_package.Config(
        max_num_batched_tokens=4,
        max_num_seqs=2,
        kvcache_block_size=4,
        num_kvcache_blocks=12,
    )
    candidate_engine = candidate_package.LLMEngine(config)
    reference_engine = RefLLMEngine(RefConfig(**config.__dict__))

    seed_prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    candidate_seed = sequence_module.Sequence(seed_prompt, candidate_package.SamplingParams(max_tokens=1))
    candidate_engine.scheduler.block_manager.allocate(candidate_seed)
    candidate_engine.scheduler.block_manager.deallocate(candidate_seed)

    reference_seed = RefSequence(seed_prompt, RefSamplingParams(max_tokens=1))
    reference_engine.scheduler.block_manager.allocate(reference_seed)
    reference_engine.scheduler.block_manager.deallocate(reference_seed)

    shared_prompts = [
        [1, 2, 3, 4, 5, 6, 7, 8, 21, 22],
        [1, 2, 3, 4, 5, 6, 7, 8, 31, 32],
    ]

    try:
        candidate_outputs = candidate_engine.generate(shared_prompts, candidate_package.SamplingParams(max_tokens=2))
        candidate_detail = f"num_outputs={len(candidate_outputs)} lengths={[len(item['token_ids']) for item in candidate_outputs]}"
    except Exception as exc:
        candidate_outputs = None
        candidate_detail = f"exception={type(exc).__name__}: {exc}"

    reference_outputs = reference_engine.generate(shared_prompts, RefSamplingParams(max_tokens=2))
    reference_detail = f"num_outputs={len(reference_outputs)} lengths={[len(item['token_ids']) for item in reference_outputs]}"

    passed = (
        candidate_outputs is not None
        and len(candidate_outputs) == len(reference_outputs) == 2
        and [len(item["token_ids"]) for item in candidate_outputs] == [2, 2]
    )
    return passed, f"candidate({candidate_detail}) reference({reference_detail})"


def run_hidden_suite(candidate_repo: Path) -> dict:
    candidate_package, scheduler_module, sequence_module = _load_candidate_modules(candidate_repo)
    cases = [
        ("single_shared_prefix_prefill", 3, lambda: _run_shared_prefix_single(candidate_package, scheduler_module, sequence_module)),
        ("batched_shared_prefix_prefill", 4, lambda: _run_shared_prefix_batch(candidate_package, scheduler_module, sequence_module)),
        ("end_to_end_shared_prefix_generation", 3, lambda: _run_end_to_end(candidate_package, sequence_module)),
    ]

    results = []
    earned_weight = 0
    total_weight = 0

    for name, weight, runner in cases:
        try:
            passed, detail = runner()
        except Exception as exc:
            passed = False
            detail = f"uncaught_exception={type(exc).__name__}: {exc}"
        results.append(CaseResult(name=name, weight=weight, passed=passed, detail=detail))
        total_weight += weight
        if passed:
            earned_weight += weight

    return {
        "score": earned_weight / total_weight if total_weight else 0.0,
        "earned_weight": earned_weight,
        "total_weight": total_weight,
        "cases": [item.__dict__ for item in results],
    }
