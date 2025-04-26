from typing import List


class LlamaVocab:
    def __init__(self, vocab):
        self._vocab = vocab

    def tokenize(self, text: str, add_bos: bool = False, special: bool = True) -> List[int]:
        return self._vocab.tokenize(text, add_bos, special)

    def detokenize(self, tokens: List[int]) -> str:
        return self._vocab.detokenize(tokens)

    def token_to_piece(self, token: int) -> str:
        return self._vocab.token_to_piece(token)

    @property
    def bos_token(self) -> int:
        return self._vocab.bos_token

    @property
    def eos_token(self) -> int:
        return self._vocab.eos_token

    @property
    def n_tokens(self) -> int:
        return self._vocab.n_tokens
