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
        model_name="openai/gpt-oss-120b:free",
        api_key=None,
        debug=False,
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
            stop=stop_after_attempt(10),
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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return response.choices[0].message.content
