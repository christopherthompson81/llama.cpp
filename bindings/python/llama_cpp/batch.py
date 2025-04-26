from typing import List, Union
import numpy as np
from . import llama_cpp


class LlamaBatch:
    def __init__(self, n_tokens: int, embd: int = 0, n_seq_max: int = 1):
        self._batch = llama_cpp.LlamaBatch(n_tokens, embd, n_seq_max)

    @property
    def tokens(self) -> np.ndarray:
        return self._batch.tokens

    @tokens.setter
    def tokens(self, value: Union[List[int], np.ndarray]) -> None:
        tokens_array = np.array(value, dtype=np.int32)
        self._batch.tokens = tokens_array

    @property
    def positions(self) -> np.ndarray:
        return self._batch.positions

    @positions.setter
    def positions(self, value: Union[List[int], np.ndarray]) -> None:
        self._batch.positions = np.array(value, dtype=np.int32)

    @property
    def n_seq_id(self) -> np.ndarray:
        return self._batch.n_seq_id

    @n_seq_id.setter
    def n_seq_id(self, value: Union[List[int], np.ndarray]) -> None:
        self._batch.n_seq_id = np.array(value, dtype=np.int32)

    @property
    def logits(self) -> np.ndarray:
        return self._batch.logits

    @logits.setter
    def logits(self, value: Union[List[int], np.ndarray]) -> None:
        self._batch.logits = np.array(value, dtype=np.int8)
