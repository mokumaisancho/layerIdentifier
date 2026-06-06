"""MLX-backed conjecture client for Apple Silicon.

Replaces Ollama LLMClient with MLX-LM for Qwen 3.5-4B generation.
Handles thinking-block stripping (Qwen 3.5 defaults to thinking mode
with no API-level disable).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from .capture import install_captures, restore_layers
from .llm import (
    ConjectureResponse,
    _extract_from_response,
    _sanitize_predicate,
)

# Thinking block patterns produced by Qwen 3.5 via MLX-LM
_THINKING_BLOCK_RE = re.compile(
    r"<think\b[^>\n]*[\n>].*?</think\b[^>\n]*[\n>]", re.DOTALL
)
_THINKING_PROCESS_RE = re.compile(r"^Thinking Process:.*?={3,}\s*", re.DOTALL)


class MLXConjectureClient:
    """MLX-LM backend for conjecture generation on Apple Silicon.

    Handles thinking-block stripping for Qwen 3.5 models which
    default to thinking mode with no way to disable via MLX-LM API.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3.5-4B-4bit",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._model: Any = None
        self._tokenizer: Any = None
        self._last_capture: dict[int, np.ndarray] | None = None

    @property
    def last_capture(self) -> dict[int, np.ndarray] | None:
        """Hidden states captured during last generation. Maps layer_idx -> ndarray."""
        return self._last_capture

    @property
    def model(self) -> str:
        """Model name/identifier (Ollama-compatible property)."""
        return self.model_name

    # ------------------------------------------------------------------
    # Ollama-compatible chat interface (used by abduction engine)
    # ------------------------------------------------------------------

    def _ensure_client(self) -> MLXConjectureClient:
        """Return self as the chat client (Ollama-compatible interface)."""
        return self

    def chat(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ollama-compatible chat() wrapping MLX generation.

        Parameters
        ----------
        model : str, optional
            Ignored (we use self.model_name).
        messages : list[dict]
            Chat messages in Ollama format: [{"role": "user", "content": "..."}].
        options : dict, optional
            May contain ``num_predict`` for max tokens.

        Returns
        -------
        dict in Ollama format: ``{"message": {"content": str}}``.
        """
        max_tokens = None
        if options and "num_predict" in options:
            max_tokens = options["num_predict"]
        raw = self._generate_raw(messages, max_tokens=max_tokens)
        return {"message": {"content": raw}}

    # ------------------------------------------------------------------
    # Thinking-block stripping
    # ------------------------------------------------------------------

    def _strip_thinking(self, text: str) -> str:
        """Remove thinking blocks from Qwen 3.5 output."""
        text = _THINKING_BLOCK_RE.sub("", text)
        text = _THINKING_PROCESS_RE.sub("", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> tuple[Any, Any]:
        """Lazy-load model. Loads once, stays in memory (~2.5GB)."""
        if self._model is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_name)
        return self._model, self._tokenizer

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if MLX-LM is available."""
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Prompt building (mirrors LLMClient.generate_conjecture template)
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        domain: str,
        history: list[dict[str, Any]] | None = None,
        constraint: str | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for conjecture generation.

        Uses the same prompt structure as LLMClient.generate_conjecture()
        to maintain AC2.3 compatibility.
        """
        history_text = ""
        if history:
            lines: list[str] = []
            for h in history[-5:]:
                status = h.get("status", "unknown")
                stmt = h.get("statement", "?")
                if status == "falsified":
                    ce = h.get("counterexample_value", "?")
                    lines.append(f"- FALSIFIED: {stmt} (counterexample: n={ce})")
                else:
                    lines.append(f"- {status.upper()}: {stmt}")
            history_text = "\n".join(lines)

        system_prompt = (
            "You are a mathematical conjecture generator.\n"
            "Output EXACTLY this format:\n"
            "REASONING: <brief reasoning>\n"
            "STATEMENT: <conjecture in plain English>\n"
            "PREDICATE: <safe Python expression using variable n>\n"
            "\n"
            "Available helpers (use ONLY these — no others exist):\n"
            "- is_prime(n), divisors(n), sigma(n), tau(n)\n"
            "- gcd(a,b), euler_phi(n), mobius(n)\n"
            "- is_perfect_square(n), prime_factors(n)\n"
            "- all(...), any(...), sum(...), prod(...)\n"
            "- abs(n), min(a,b), max(a,b), pow(a,b), len(x), range(n), sqrt(n)\n"
            "\n"
            "Examples of good predicates:\n"
            "- is_prime(n**2 + n + 41)\n"
            "- sigma(n) < 2 * n\n"
            "- tau(n) % 2 == 1\n"
            "- euler_phi(n) > n / 2\n"
            "- all(is_prime(d) for d in divisors(n))\n"
        )

        history_section = (
            "Previous attempts (do NOT repeat these):\n" + history_text
            if history_text
            else "This is your first attempt."
        )

        constraint_section = (
            "\nADDITIONAL CONSTRAINT: " + constraint if constraint else ""
        )

        user_prompt = (
            f"You are exploring mathematical properties of {domain}.\n"
            f"Generate a novel conjecture — a statement about integers "
            f"that you think might be true.\n"
            f"\n"
            f"{history_section}\n"
            f"{constraint_section}\n"
            f"\n"
            f"Be specific. Avoid trivial statements. "
            f"Use ONLY the helpers listed above."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Raw MLX generation
    # ------------------------------------------------------------------

    def _generate_raw(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> str:
        """Call MLX-LM and return raw text with thinking stripped.

        Also captures hidden states from layers 0, 15, 31 into
        self._last_capture when the model exposes the expected layer
        structure.
        """
        model, tokenizer = self._ensure_loaded()
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Attempt hidden-state capture
        self._last_capture = None
        wrappers: dict | None = None
        originals: dict | None = None

        try:
            wrappers, originals = install_captures(model)
            self._last_capture = {}
        except (AttributeError, IndexError):
            # Model doesn't have expected layer structure — capture not available
            wrappers = None
            originals = None

        try:
            if wrappers is not None:
                # Use generate_step for capture path
                from mlx_lm.generate import generate_step
                from mlx_lm.sample_utils import make_sampler

                import mlx.core as mx

                prompt_ids = tokenizer.encode(prompt)
                prompt_arr = mx.array(prompt_ids)
                sampler = make_sampler(self._temperature)

                generated_ids: list[int] = []
                for token, _logprobs in generate_step(
                    prompt_arr,
                    model,
                    max_tokens=max_tokens or self._max_tokens,
                    sampler=sampler,
                ):
                    token_id = token.item() if hasattr(token, "item") else int(token)
                    generated_ids.append(token_id)
                    eos_ids = (
                        getattr(tokenizer, "eos_token_ids", None)
                        or {getattr(tokenizer, "eos_token_id", -1)}
                    )
                    if token_id in eos_ids:
                        break

                text = tokenizer.decode(generated_ids)

                # Collect hidden states from wrappers (skip index 0 = prefill)
                for layer_idx, w in wrappers.items():
                    if len(w.hidden_states) > 1:
                        self._last_capture[layer_idx] = np.stack(
                            w.hidden_states[1:]
                        )
                    else:
                        self._last_capture[layer_idx] = np.array([])
            else:
                # Fallback: use high-level generate (no capture)
                from mlx_lm import generate as mlx_generate

                text = mlx_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens or self._max_tokens,
                    temp=self._temperature,
                    verbose=False,
                )
        finally:
            if originals is not None:
                restore_layers(model, originals)

        return self._strip_thinking(text)

    # ------------------------------------------------------------------
    # Public API (mirrors LLMClient interface)
    # ------------------------------------------------------------------

    def generate_conjecture(
        self,
        domain: str,
        history: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        constraint: str | None = None,
    ) -> ConjectureResponse:
        """Generate a novel conjecture for the given domain."""
        messages = self._build_messages(domain, history, constraint)
        raw = self._generate_raw(messages, max_tokens)
        response = _extract_from_response(raw)
        if response.predicate_source:
            response.predicate_source = _sanitize_predicate(
                response.predicate_source
            )
        return response

    def refine_conjecture(
        self,
        statement: str,
        predicate_source: str,
        counterexample_value: int | None = None,
        *,
        max_tokens: int | None = None,
    ) -> ConjectureResponse:
        """Refine a falsified conjecture given its counterexample."""
        messages = self._build_refine_messages(
            statement, predicate_source, counterexample_value
        )
        raw = self._generate_raw(messages, max_tokens)
        response = _extract_from_response(raw)
        if response.predicate_source:
            response.predicate_source = _sanitize_predicate(
                response.predicate_source
            )
        return response

    # ------------------------------------------------------------------
    # Refine prompt building
    # ------------------------------------------------------------------

    def _build_refine_messages(
        self,
        statement: str,
        predicate_source: str,
        counterexample_value: int | None,
    ) -> list[dict[str, str]]:
        """Build chat messages for conjecture refinement."""
        user_prompt = (
            f"Your conjecture was falsified:\n"
            f"- Statement: {statement}\n"
            f"- Predicate: {predicate_source}\n"
            f"- Counterexample: n = {counterexample_value}\n"
            f"\n"
            f"Refine the conjecture to account for this counterexample. "
            f"You can:\n"
            f"1. Restrict the domain (e.g., 'for odd n only')\n"
            f"2. Add a guard condition (e.g., 'for n not divisible by 3')\n"
            f"3. Modify the formula slightly\n"
            f"\n"
            f"Available helpers (use ONLY these):\n"
            f"- is_prime(n), divisors(n), sigma(n), tau(n)\n"
            f"- gcd(a,b), euler_phi(n), mobius(n)\n"
            f"- is_perfect_square(n), prime_factors(n)\n"
            f"- abs(n), min(a,b), max(a,b), pow(a,b), len(x)\n"
            f"\n"
            f"Output EXACTLY:\n"
            f"REASONING: <brief reasoning about the refinement>\n"
            f"STATEMENT: <refined conjecture>\n"
            f"PREDICATE: <safe Python expression using variable n>"
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a mathematical conjecture refiner.\n"
                    "Output EXACTLY:\n"
                    "REASONING: <brief reasoning>\n"
                    "STATEMENT: <refined conjecture>\n"
                    "PREDICATE: <safe Python expression using variable n>"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
