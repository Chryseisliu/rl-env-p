from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


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
        if block.ref_count != 0:
            raise ValueError("block is not free")
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        if self.blocks[block_id].ref_count != 0:
            raise ValueError("block is still referenced")
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
        if seq.block_table:
            raise ValueError("sequence already allocated")

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
            if last_block.hash == -1:
                raise ValueError("expected a finalized block before append allocation")
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            if last_block.hash != -1:
                raise ValueError("expected an open block at block boundary")
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            block_hash = self.compute_hash(token_ids, prefix)
            last_block.update(block_hash, token_ids)
            self.hash_to_block_id[block_hash] = last_block.block_id
        else:
            if last_block.hash != -1:
                raise ValueError("expected an open block while appending within block")
