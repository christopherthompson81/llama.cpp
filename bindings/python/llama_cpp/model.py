from typing import Dict, List, Optional, TYPE_CHECKING

from . import llama_cpp
from . import _LlamaModel
from .batch import LlamaBatch

if TYPE_CHECKING:
    from .context import LlamaContext


class LlamaModel:
    """High-level Python wrapper for llama.cpp model."""

    def __init__(self, model_path: str):
        """
        Initialize a LlamaModel.

        Args:
            model_path: Path to the GGUF model file
        """
        self._model = _LlamaModel(model_path)
        self._vocab = self._model.vocab

    def create_context(self, **kwargs) -> 'LlamaContext':
        """
        Create a new context for inference.

        Args:
            **kwargs: Context parameters (n_ctx, n_batch, n_threads, etc.)

        Returns:
            A new LlamaContext instance
        """
        from .context import LlamaContext
        return LlamaContext(self, kwargs)

    def create_batch(self, n_tokens: int, embd: int = 0, n_seq_max: int = 1) -> LlamaBatch:
        """
        Create a new batch for token processing.

        Args:
            n_tokens: Maximum number of tokens in the batch
            embd: Embedding dimension (0 for token-based batch)
            n_seq_max: Maximum number of sequences

        Returns:
            A new LlamaBatch instance
        """
        return LlamaBatch(n_tokens, embd, n_seq_max)

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

    def apply_chat_template(self,
                            messages: List[Dict[str, str]],
                            custom_template: Optional[str] = None) -> str:
        """
        Apply the chat template to a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            custom_template: Optional custom template string

        Returns:
            Formatted chat text
        """
        return llama_cpp.chat_apply_template(self._model, messages, custom_template or "")

    @property
    def n_ctx_train(self) -> int:
        """Get the context size used during training."""
        return self._model.n_ctx_train

    @property
    def n_embd(self) -> int:
        """Get the embedding size."""
        return self._model.n_embd

    @property
    def n_layer(self) -> int:
        """Get the number of layers."""
        return self._model.n_layer

    @property
    def n_head(self) -> int:
        """Get the number of attention heads."""
        return self._model.n_head

    @property
    def n_head_kv(self) -> int:
        """Get the number of key/value heads (for grouped-query attention)."""
        return self._model.n_head_kv

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab.n_tokens

    @property
    def bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self._vocab.bos_token

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self._vocab.eos_token

    def get_metadata(self) -> Dict[str, str]:
        """
        Get model metadata.

        Returns:
            Dictionary of metadata key-value pairs
        """
        # Common metadata keys
        keys = [
            "general.name",
            "general.family",
            "general.architecture",
            "general.description",
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.feed_forward_length",
            "llama.rope.dimension_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.attention.layer_norm_rms_epsilon",
            "llama.rope.freq_base",
        ]

        metadata = {}
        for key in keys:
            value = self._model.meta_val(key)
            if value:
                metadata[key] = value

        return metadata
