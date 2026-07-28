"""
openrouter_agent.py
Agent for calling models via the OpenRouter API with retry + exponential backoff.
"""

import os
import openai
from tenacity import (
    Retrying,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from agents.base_llm_agent import BaseLLMAgent


class OpenRouterAgent(BaseLLMAgent):
    def __init__(
        self,
        player_id,
        model_name="openai/gpt-oss-120b",
        api_key=None,
        debug=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(player_id, debug=debug)

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY "
                "environment variable or pass api_key= to constructor."
            )

        self.model_name = model_name
        self.verbose = verbose
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        print(
            f"[{self.player_id}] Initializing OpenRouter client (model={model_name})..."
        )

    def _generate_llm_response(self, prompt):
        for attempt in Retrying(
            wait=wait_exponential(multiplier=2, min=2, max=1024),
            stop=stop_after_attempt(6),
            retry=retry_if_exception_type(
                (
                    openai.APIError,
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.InternalServerError,
                )
            ),
        ):
            with attempt:
                if self.verbose and attempt.retry_state.outcome is not None:
                    print(
                        f"[P{self.player_id} OpenRouter] "
                        f"Retry {attempt.retry_state.attempt_number - 1}/10 "
                        f"(waited {attempt.retry_state.idle_for:.1f}s) "
                        f"after: {attempt.retry_state.outcome.exception()}"
                    )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    extra_body={"provider": {"quantizations": ["fp4"]}},
                )
                if self.verbose:
                    u = response.usage
                    if u:
                        print(
                            f"[P{self.player_id} OpenRouter] "
                            f"Tokens: prompt={u.prompt_tokens} "
                            f"completion={u.completion_tokens} "
                            f"total={u.total_tokens}"
                        )
                return response.choices[0].message.content
