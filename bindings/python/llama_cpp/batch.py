from typing import List, Optional, Union
import numpy as np

from . import llama_cpp

class LlamaBatch:
    """High-level Python wrapper for llama.cpp batch."""
    
    def __init__(self, n_tokens: int, embd: int = 0, n_seq_max: int = 1):
        """
        Initialize a LlamaBatch.
        
        Args:
            n_tokens: Maximum number of tokens in the batch
            embd: Embedding size (0 for token-based batch)
            n_seq_max: Maximum number of sequences
        """
        self._batch = llama_cpp.LlamaBatch(n_tokens, embd, n_seq_max)
    
    @property
    def tokens(self) -> np.ndarray:
        """Get the token IDs in the batch."""
        return self._batch.tokens
    
    @tokens.setter
    def tokens(self, value: Union[List[int], np.ndarray]) -> None:
        """
        Set the token IDs in the batch.
        
        Args:
            value: Array of token IDs
        """
        tokens_array = np.array(value, dtype=np.int32)
        
        # Get the batch capacity
        batch = self._batch.get_batch()
        
        # Check if the array is too large before setting
        if tokens_array.size > batch.n_tokens:
            raise ValueError(f"Token array size ({tokens_array.size}) exceeds batch capacity ({batch.n_tokens})")
            
        # Use the property setter from the C++ binding
        self._batch.tokens = tokens_array
    
    @property
    def positions(self) -> np.ndarray:
        """Get the positions in the batch."""
        return self._batch.positions
    
    @positions.setter
    def positions(self, value: Union[List[int], np.ndarray]) -> None:
        """
        Set the positions in the batch.
        
        Args:
            value: Array of positions
        """
        self._batch.positions = np.array(value, dtype=np.int32)
    
    @property
    def n_seq_id(self) -> np.ndarray:
        """Get the sequence IDs in the batch."""
        return self._batch.n_seq_id
    
    @n_seq_id.setter
    def n_seq_id(self, value: Union[List[int], np.ndarray]) -> None:
        """
        Set the sequence IDs in the batch.
        
        Args:
            value: Array of sequence IDs
        """
        self._batch.n_seq_id = np.array(value, dtype=np.int32)
    
    @property
    def logits(self) -> np.ndarray:
        """Get the logits flags in the batch."""
        return self._batch.logits
    
    @logits.setter
    def logits(self, value: Union[List[int], np.ndarray]) -> None:
        """
        Set the logits flags in the batch.
        
        Args:
            value: Array of logits flags (0 or 1)
        """
        self._batch.logits = np.array(value, dtype=np.int8)
