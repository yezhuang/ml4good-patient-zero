"""Tinker LoRA-checkpoint sampling client.

Tinker is not an OpenAI-compatible HTTP endpoint. A fine-tuned checkpoint is
loaded through the `tinker` SDK and sampled locally:

    service_client = tinker.ServiceClient()
    tc = service_client.create_lora_training_client(base_model=..., rank=...)
    tc.load_state("tinker://...")
    sampler = tc.save_weights_and_get_sampling_client(name=...)

This client wraps that flow behind the same `generate(system_prompt,
user_prompt) -> GenerationResult` interface used by `OpenAICompatibleClient`,
so `ModelAgent` can drive it unchanged.

Authentication uses the `tinker` SDK's own env handling (`TINKER_API_KEY`);
there is no API key to pass explicitly.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("ml4good.tinker")

from src.models.openai_compatible import GenerationResult

# The role_colon renderer's only stop is "\n\nUser:", but base-SFT checkpoints
# often run on past their turn, hallucinating a fake transcript whose stray tokens
# corrupt parsing. They do this two ways: emitting a pretraining-style "Human:"
# boundary, or replaying the env's own structural markers ("[GAME]", "[Player N]").
# Stop on all of these so a turn ends at the model's next-speaker marker. The
# "\n" prefixes ensure legitimate decision tokens like "[0 defect]" are not clipped.
DEFAULT_EXTRA_STOP = [
    # Bare conversational role markers (a colon makes them unambiguous turn
    # boundaries). Bare form catches space- AND newline-prefixed leaks like
    # "...both.  Human:" that "\n\nHuman:" missed, and never occurs inside a
    # decision token such as "[0 defect]".
    "Human:",
    "Assistant:",
    "System:",
    "User:",
    # "[GAME]" can be bare — it never appears inside a decision token like
    # "[0 defect]" — which also catches the space-prefixed " [GAME]" ramble leak.
    "[GAME]",
    # "[Player" must stay prefixed (newline or space): a model may legitimately
    # lead its decision with a "[Player N]" tag, so a bare stop would clip it empty.
    "\n\n[Player",
    "\n[Player",
    " [Player",
]

# Loading a checkpoint and building a renderer are expensive, so cache them
# across agents within a process. Two agents sharing a checkpoint reuse one
# sampler.
_SAMPLER_CACHE: dict[str, Any] = {}
_RENDERER_CACHE: dict[tuple[str, str], Any] = {}
_CACHE_LOCK = threading.Lock()

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B-Base"
# Fallback only. By default the renderer is resolved from the base model via
# tinker_cookbook.model_info.get_recommended_renderer_name (see _get_renderer),
# so sampling matches what training used. For Qwen3-8B-Base that is role_colon.
FALLBACK_RENDERER = "role_colon"


def truncate_at_stops(text: str, stops: list[str]) -> str:
    """Cut `text` at the earliest occurrence of any stop sequence, then rstrip."""
    cut = len(text)
    for seq in stops:
        idx = text.find(seq)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()


@dataclass
class TinkerClient:
    state_path: str
    base_model: str = DEFAULT_BASE_MODEL
    rank: int = 32
    # None → resolve from base_model so sampling matches training.
    renderer_name: str | None = None
    temperature: float = 0.2
    max_tokens: int = 64
    top_p: float = 1.0
    extra_stop: list[str] = field(default_factory=lambda: list(DEFAULT_EXTRA_STOP))
    # Retry transient sampling failures (service hiccups, rate limits) so batches
    # run safely under concurrency. Exponential backoff with jitter, like the
    # OpenAI-compatible client.
    max_retries: int = 5

    @property
    def model(self) -> str:
        return self.state_path

    def _get_sampler(self) -> Any:
        with _CACHE_LOCK:
            if self.state_path not in _SAMPLER_CACHE:
                import tinker

                service_client = tinker.ServiceClient()
                tc = service_client.create_lora_training_client(
                    base_model=self.base_model,
                    rank=self.rank,
                )
                tc.load_state(self.state_path)
                _SAMPLER_CACHE[self.state_path] = tc.save_weights_and_get_sampling_client()
        return _SAMPLER_CACHE[self.state_path]

    def _resolve_renderer_name(self) -> str:
        """Match the renderer used at training time for this base model."""
        if self.renderer_name:
            return self.renderer_name
        try:
            from tinker_cookbook import model_info

            return model_info.get_recommended_renderer_name(self.base_model)
        except Exception:
            return FALLBACK_RENDERER

    def _get_renderer(self) -> Any:
        renderer_name = self._resolve_renderer_name()
        key = (renderer_name, self.base_model)
        with _CACHE_LOCK:
            if key not in _RENDERER_CACHE:
                from tinker_cookbook import renderers, tokenizer_utils

                tokenizer = tokenizer_utils.get_tokenizer(self.base_model)
                _RENDERER_CACHE[key] = renderers.get_renderer(renderer_name, tokenizer)
        return _RENDERER_CACHE[key]

    def _sample_with_retry(self, sampler: Any, model_input: Any, sampling_params: Any) -> Any:
        """Sample with exponential backoff. The Tinker SDK does not expose typed
        transient errors, so any exception is retried up to max_retries (capped
        backoff), then re-raised — a persistent fault still surfaces."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return sampler.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                ).result()
            except Exception as exc:  # noqa: BLE001 - SDK errors are untyped
                last_error = exc
                if attempt < self.max_retries:
                    delay = min(2 ** attempt, 30) + random.uniform(0, 0.5)
                    logger.warning(
                        "tinker sample failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, exc, delay,
                    )
                    time.sleep(delay)
        raise last_error if last_error else RuntimeError("tinker sampling failed")

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import tinker
        from tinker_cookbook import renderers

        sampler = self._get_sampler()
        renderer = self._get_renderer()

        stop = list(renderer.get_stop_sequences())
        for seq in self.extra_stop:
            if seq not in stop:
                stop.append(seq)

        sampling_params = tinker.types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
        )

        messages = []
        if system_prompt:
            messages.append(renderers.Message(role="system", content=system_prompt))
        messages.append(renderers.Message(role="user", content=user_prompt))
        model_input = renderer.build_generation_prompt(messages)

        result = self._sample_with_retry(sampler, model_input, sampling_params)
        tokens = result.sequences[0].tokens
        text = renderer.parse_response(tokens)[0]["content"]
        # The stop sequence that halted generation can be left in the decoded text
        # (e.g. a trailing "[GAME]" or "Assistant:"). Cut at the earliest stop so the
        # transcript and any downstream parsing see only the model's real content.
        text = truncate_at_stops(text, stop)

        raw = {
            "backend": "tinker",
            "state_path": self.state_path,
            "base_model": self.base_model,
            "renderer": self._resolve_renderer_name(),
            "num_output_tokens": len(tokens),
        }
        return GenerationResult(text=text, raw=raw)
