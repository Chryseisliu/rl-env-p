# Nano-vLLM Engine Single-Task RL Environment

I wrote this prototype around the nano-vllm engine layer from `GeeeekExplorer/nano-vllm` rather than around my earlier custom mini-decoder. If I want a single-task RL environment that is still representative of modern LLM inference, this is where the interesting control-flow problems live: request scheduling, prefix caching, block ownership, cache pressure, and decode lifecycle.

I intentionally did not try to vendor the entire upstream project. For this environment, I only need the engine slice and a lightweight deterministic runner. That keeps the task judgeable and self-contained while still making it feel like real modern inference-system work instead of a toy exercise.

## Upstream Basis

This prototype is adapted from the MIT-licensed `nano-vllm` project:

- Repository: `https://github.com/GeeeekExplorer/nano-vllm`
- Engine subtree used as the code basis: `nanovllm/engine`

## What The Task Is

The task is to repair a bug in an adapted local copy of the `nanovllm` engine implementation.

The bug is in prefill admission under prefix caching. The baseline engine correctly stores reusable full prompt blocks, but its scheduler still budgets prefill work using total prompt length instead of uncached prompt length. Under tight `max_num_batched_tokens`, that causes the engine to reject or stall requests even when the cache should allow them to proceed.

I like this version much more because it is closer to the real shape of modern inference systems work:

- the solver has to reason about the scheduler and block manager together
- the bug only appears under realistic shared-prefix workloads
- the judge can stay deterministic and behavioral
- the environment is still small enough to understand in one sitting

## Layout

- `TASK_PROMPT.md`: the formal task statement I would hand to the agent.
- `task_repo_baseline/`: the buggy adapted local `nanovllm` package the agent is allowed to modify.
- `judge_assets/`: the trusted evaluation side, including hidden tests and a corrected reference implementation.
- `simple_examples/`: example patches that show the evaluator working end to end.
- `run_eval.py`: the clean-snapshot evaluator.
- `THIRD_PARTY_NOTICES.md`: attribution for the upstream engine source I adapted.

## Intended Environment

I designed this environment assuming the evaluation image preinstalls the following packages:

- `python>=3.10`
- `torch`
- `numpy`
- `pytest`
- `transformers`
- `tqdm`
- `xxhash`
- `typing_extensions`

## Evaluation Flow

(For the judge) We want to evaluate patches, not a dirty workspace.

1. restore a fresh copy of `task_repo_baseline/`.
2. apply only the candidate patch.
3. reject patches that touch protected evaluation assets or public tests.
4. run public tests in the candidate repo.
5. run hidden behavioral tests from the trusted side in a fresh Python process.

The score is about engine behavior, not code style. I care about whether the candidate restores correct scheduling semantics under prefix caching pressure.

The public tests intentionally expose the broad shape of the problem without fully specifying the fix. They show that prefix blocks are reusable and that shared-prefix prefill should still make progress under a tight token budget. The hidden tests then check the more complete scheduler behavior.

## Quickstart

You can run the evaluator against the included example patches with:

```bash
python3 run_eval.py simple_examples/example_bad.patch
python3 run_eval.py simple_examples/example_good.patch
```

## Reproducible Eval Image

I also packaged the prototype as a simple Docker image so the intended environment can be recreated directly instead of being inferred from prose.

You can build the image with:

```bash
docker build -t nano-vllm-rl-env .
```

You can run the evaluator inside that image with:

```bash
docker run --rm -it nano-vllm-rl-env python3 run_eval.py simple_examples/example_bad.patch
docker run --rm -it nano-vllm-rl-env python3 run_eval.py simple_examples/example_good.patch
```

If you want to iterate on candidate fixes from my local checkout, you can mount the workspace into the container:

```bash
docker run --rm -it -v "$PWD":/workspace -w /workspace nano-vllm-rl-env bash
```
