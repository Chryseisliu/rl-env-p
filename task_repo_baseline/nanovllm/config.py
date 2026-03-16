from dataclasses import dataclass


@dataclass
class Config:
    max_num_batched_tokens: int = 16
    max_num_seqs: int = 4
    max_model_len: int = 128
    eos: int = -1
    kvcache_block_size: int = 4
    num_kvcache_blocks: int = 32

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.kvcache_block_size <= 0:
            raise ValueError("kvcache_block_size must be positive")
        if self.num_kvcache_blocks <= 0:
            raise ValueError("num_kvcache_blocks must be positive")
