# Task Prompt

I frame the agent task this way:

> Your task is to repair a bug in an adapted local copy of the `nanovllm` inference engine.
>
> The repository implements request scheduling, prefix caching, KV-cache block management, and batched autoregressive generation. The current implementation is behaviorally incorrect in some realistic shared-prefix workloads. In particular, the scheduler does not correctly account for reusable cached prompt blocks when deciding whether a prefill batch fits within the token budget.
>
> You may inspect and modify repository files, run tests, and create your own debugging scripts. Your goal is to make engine behavior correct.
>
> Requirements:
>
> - Do not modify the test harness or evaluation scripts.
> - Do not hardcode outputs for specific test inputs.
> - Preserve the public API expected by the repository.
> - Your final solution must correctly handle prefix caching under shared-prefix prompts.
> - Your final solution must remain correct under constrained `max_num_batched_tokens`.
> - Your final solution must preserve scheduler and block-manager invariants.
> - You may add helper functions or internal tests, but the final repository must pass the hidden evaluation suite.
>
> Deliverable:
>
> Modify the repository so that the engine behaves correctly. The judge will evaluate your final code by running a hidden behavioral test suite against the repository.

I chose this wording because it points the agent at the real system behavior without giving away the exact fix.
