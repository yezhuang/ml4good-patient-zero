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

from dataclasses import dataclass
from typing import Any

from src.models.openai_compatible import GenerationResult

# Loading a checkpoint and building a renderer are expensive, so cache them
# across agents within a process. Two agents sharing a checkpoint reuse one
# sampler.
_SAMPLER_CACHE: dict[str, Any] = {}
_RENDERER_CACHE: dict[tuple[str, str], Any] = {}

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B-Base"
# Fallback only. By default the renderer is resolved from the base model via
# tinker_cookbook.model_info.get_recommended_renderer_name (see _get_renderer),
# so sampling matches what training used. For Qwen3-8B-Base that is role_colon.
FALLBACK_RENDERER = "role_colon"


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

    @property
    def model(self) -> str:
        return self.state_path

    def _get_sampler(self) -> Any:
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
        if key not in _RENDERER_CACHE:
            from tinker_cookbook import renderers, tokenizer_utils

            tokenizer = tokenizer_utils.get_tokenizer(self.base_model)
            _RENDERER_CACHE[key] = renderers.get_renderer(renderer_name, tokenizer)
        return _RENDERER_CACHE[key]

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import tinker
        from tinker_cookbook import renderers

        sampler = self._get_sampler()
        renderer = self._get_renderer()

        sampling_params = tinker.types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=renderer.get_stop_sequences(),
        )

        messages = []
        if system_prompt:
            messages.append(renderers.Message(role="system", content=system_prompt))
        messages.append(renderers.Message(role="user", content=user_prompt))
        model_input = renderer.build_generation_prompt(messages)

        result = sampler.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()
        tokens = result.sequences[0].tokens
        text = renderer.parse_response(tokens)[0]["content"]

        raw = {
            "backend": "tinker",
            "state_path": self.state_path,
            "base_model": self.base_model,
            "renderer": self._resolve_renderer_name(),
            "num_output_tokens": len(tokens),
        }
        return GenerationResult(text=text, raw=raw)
