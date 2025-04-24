from typing import List, Optional
import numpy as np

from . import llama_cpp, _LlamaSampler

class LlamaSampler:
    """High-level Python wrapper for llama.cpp sampler."""
    
    def __init__(self):
        """Initialize a LlamaSampler."""
        self._sampler = _LlamaSampler()
    
    def add_top_k(self, k: int) -> None:
        """
        Add top-k sampling.
        
        Args:
            k: The k value (number of tokens to consider)
        """
        self._sampler.add_top_k(k)
    
    def add_top_p(self, p: float) -> None:
        """
        Add top-p (nucleus) sampling.
        
        Args:
            p: The p value (probability threshold)
        """
        self._sampler.add_top_p(p)
    
    def add_temperature(self, temperature: float) -> None:
        """
        Add temperature sampling.
        
        Args:
            temperature: The temperature value
        """
        self._sampler.add_temperature(temperature)
    
    def add_mirostat(self, tau: float, eta: float, m: int) -> None:
        """
        Add mirostat sampling (adaptive temperature).
        
        Args:
            tau: Target entropy
            eta: Learning rate
            m: Order of mirostat (1 or 2)
        """
        self._sampler.add_mirostat(tau, eta, m)
    
    def add_grammar(self, grammar: str) -> None:
        """
        Add grammar-based sampling.
        
        Args:
            grammar: Grammar definition string
        """
        self._sampler.add_grammar(grammar)
    
    def sample(self, ctx, last_tokens: List[int]) -> int:
        """
        Sample the next token.
        
        Args:
            ctx: The LlamaContext object
            last_tokens: Recent token history
            
        Returns:
            The sampled token ID
        """
        # Get the raw context pointer
        ctx_ptr = self._sampler.get_context_ptr(ctx)
        return self._sampler.sample(ctx_ptr, last_tokens)
    
    def accept(self, token: int) -> None:
        """
        Accept a token (for stateful samplers like mirostat).
        
        Args:
            token: The token ID that was selected
        """
        self._sampler.accept(token)
