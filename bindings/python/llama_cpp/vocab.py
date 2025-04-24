from typing import List, Optional
import numpy as np

from . import llama_cpp, _LlamaVocab

class LlamaVocab:
    """High-level Python wrapper for llama.cpp vocabulary."""
    
    def __init__(self, vocab):
        """
        Initialize a LlamaVocab.
        
        Args:
            vocab: The llama_cpp.LlamaVocab object
        """
        self._vocab = vocab
    
    def tokenize(self, text: str, add_bos: bool = False, special: bool = True) -> List[int]:
        """
        Tokenize text.
        
        Args:
            text: The text to tokenize
            add_bos: Whether to add the beginning-of-sequence token
            special: Whether to encode special tokens
        
        Returns:
            List of token IDs
        """
        return self._vocab.tokenize(text, add_bos, special)
    
    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            Decoded text
        """
        return self._vocab.detokenize(tokens)
    
    def token_to_piece(self, token: int) -> str:
        """
        Convert a single token to its string representation.
        
        Args:
            token: Token ID
        
        Returns:
            String representation of the token
        """
        return self._vocab.token_to_piece(token)
    
    @property
    def bos_token(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self._vocab.bos_token
    
    @property
    def eos_token(self) -> int:
        """Get the end-of-sequence token ID."""
        return self._vocab.eos_token
    
    @property
    def n_tokens(self) -> int:
        """Get the vocabulary size."""
        return self._vocab.n_tokens
