from typing import Dict, List, Optional, TYPE_CHECKING

from . import llama_cpp
from . import _LlamaModel
from .batch import LlamaBatch

if TYPE_CHECKING:
    from .context import LlamaContext


class LlamaModel:
    """High-level Python wrapper for llama.cpp model."""

    def __init__(self, model_path: str):
        self._model = _LlamaModel(model_path)
        self._vocab = self._model.vocab

    def create_context(self, **kwargs) -> 'LlamaContext':
        from .context import LlamaContext
        return LlamaContext(self, kwargs)

    def create_batch(self, n_tokens: int, embd: int = 0, n_seq_max: int = 1) -> LlamaBatch:
        return LlamaBatch(n_tokens, embd, n_seq_max)

    def tokenize(self, text: str, add_bos: bool = False, special: bool = True) -> List[int]:
        return self._vocab.tokenize(text, add_bos, special)

    def detokenize(self, tokens: List[int]) -> str:
        return self._vocab.detokenize(tokens)

    def token_to_piece(self, token: int) -> str:
        return self._vocab.token_to_piece(token)

    def apply_chat_template(self, messages: List[Dict[str, str]], custom_template: Optional[str] = None) -> str:
        return llama_cpp.chat_apply_template(self._model, messages, custom_template or "")

    @property
    def n_ctx_train(self) -> int:
        return self._model.n_ctx_train

    @property
    def n_embd(self) -> int:
        return self._model.n_embd

    @property
    def n_layer(self) -> int:
        return self._model.n_layer

    @property
    def n_head(self) -> int:
        return self._model.n_head

    @property
    def n_head_kv(self) -> int:
        return self._model.n_head_kv

    @property
    def vocab_size(self) -> int:
        return self._vocab.n_tokens

    @property
    def bos_token_id(self) -> int:
        return self._vocab.bos_token

    @property
    def eos_token_id(self) -> int:
        return self._vocab.eos_token

    def get_metadata(self) -> Dict[str, str]:
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
