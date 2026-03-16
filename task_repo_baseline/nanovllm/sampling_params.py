from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 4
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 1e-10:
            raise ValueError("greedy sampling is not permitted")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
