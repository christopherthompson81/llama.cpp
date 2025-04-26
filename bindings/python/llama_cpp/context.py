from typing import Dict, List, Optional, Any
import numpy as np

from .model import LlamaModel
from .batch import LlamaBatch


class LlamaContext:
    def __init__(self, model: LlamaModel, params: Dict[str, Any]):
        self._model = model

        # Set default parameters if not provided
        if 'n_ctx' not in params:
            params['n_ctx'] = 2048
        if 'n_batch' not in params:
            params['n_batch'] = 512
        if 'n_threads' not in params:
            params['n_threads'] = 0  # Auto-detect

        # Create context directly from the C++ binding
        from .llama_cpp import LlamaContext as _LlamaContext
        self._ctx = _LlamaContext(model._model, params)
        self._last_tokens = []

    def decode(self, batch: LlamaBatch) -> bool:
        return self._ctx.decode(batch._batch)

    def get_logits(self) -> np.ndarray:
        return self._ctx.get_logits()

    def get_embeddings(self) -> np.ndarray:
        return self._ctx.get_embeddings()

    def clear_kv_cache(self) -> None:
        self._ctx.kv_cache_clear()

    def set_n_threads(self, n_threads: int, n_threads_batch: Optional[int] = None) -> None:
        if n_threads_batch is None:
            n_threads_batch = n_threads
        self._ctx.set_n_threads(n_threads, n_threads_batch)

    def generate(self,
                 tokens: List[int],
                 max_tokens: int = 128,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 top_k: int = 40,
                 stop_tokens: Optional[List[int]] = None) -> List[int]:
        from .sampler import LlamaSampler

        # Create a batch for the input tokens
        batch = self._model.create_batch(len(tokens))
        batch.tokens = np.array(tokens, dtype=np.int32)
        batch.positions = np.arange(len(tokens), dtype=np.int32)
        batch.logits = np.ones(len(tokens), dtype=np.int8)
        batch.logits[-1] = 1  # Only compute logits for the last token

        # Process the input batch
        self.decode(batch)

        # Create a sampler
        sampler = LlamaSampler()
        if temperature > 0:
            sampler.add_temperature(temperature)
            if top_p < 1.0:
                sampler.add_top_p(top_p)
            if top_k > 0:
                sampler.add_top_k(top_k)

        # Keep track of all tokens
        all_tokens = tokens.copy()
        self._last_tokens = tokens.copy()

        # Generate new tokens
        for i in range(max_tokens):
            # Sample the next token
            token = sampler.sample(self._ctx, self._last_tokens[-min(len(self._last_tokens), 64):])

            # Accept the token
            sampler.accept(token)

            # Add to the result
            all_tokens.append(token)
            self._last_tokens.append(token)

            # Check for stop tokens
            if stop_tokens and token in stop_tokens:
                break

            # Create a batch for the new token
            batch = self._model.create_batch(1)
            batch.tokens = np.array([token], dtype=np.int32)
            batch.positions = np.array([len(all_tokens) - 1], dtype=np.int32)
            batch.logits = np.ones(1, dtype=np.int8)

            # Process the new token
            self.decode(batch)

        return all_tokens
