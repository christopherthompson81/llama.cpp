from typing import List
from . import _LlamaSampler


class LlamaSampler:
    def __init__(self):
        self._sampler = _LlamaSampler()

    def add_greedy(self) -> None:
        self._sampler.add_greedy()

    def add_top_k(self, k: int) -> None:
        self._sampler.add_top_k(k)

    def add_top_p(self, p: float) -> None:
        self._sampler.add_top_p(p)

    def add_temperature(self, temperature: float) -> None:
        self._sampler.add_temperature(temperature)

    def add_mirostat(self, tau: float, eta: float, m: int) -> None:
        self._sampler.add_mirostat(tau, eta, m)

    def add_grammar(self, grammar: str) -> None:
        self._sampler.add_grammar(grammar)

    def sample(self, ctx, last_tokens: List[int]) -> int:
        # Get the raw context pointer
        ctx_ptr = ctx._ctx.get_context_ptr()
        return self._sampler.sample(ctx_ptr, last_tokens)

    def accept(self, token: int) -> None:
        self._sampler.accept(token)
