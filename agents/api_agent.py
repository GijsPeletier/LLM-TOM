"""
api_agent.py
Unified Agent for Proprietary LLMs (OpenAI, Anthropic, Google).
"""

from agents.base_llm_agent import BaseLLMAgent
from tenacity import Retrying, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.genai import types
from google.genai.errors import ServerError


class APIAgent(BaseLLMAgent):
    def __init__(self, player_id, provider, model_name=None, **kwargs):
        debug = kwargs.pop("debug", False)
        super().__init__(player_id, debug=debug)  # Inherit game, history, and make_offer logic
        self.provider = provider.lower()

        print(f"[{self.player_id}] Initializing {self.provider.capitalize()} client...")

        # Dynamically initialize the correct client based on the provider
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()  # Looks for OPENAI_API_KEY env var
            self.model_name = model_name or "gpt-4o"

        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()  # Looks for ANTHROPIC_API_KEY env var
            self.model_name = model_name or "claude-3-5-sonnet-latest"

        elif self.provider == "google":
            from google import genai
            self.client = genai.Client()  # Looks for GEMINI_API_KEY env var
            self.model_name = model_name or "gemini-2.5-pro"

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _generate_llm_response(self, prompt):
        """Routes the prompt to the appropriate provider SDK."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=21000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.provider == "google":
            # Inline exponential backoff applied only to Google
            for attempt in Retrying(
                    wait=wait_exponential(multiplier=2, min=4, max=128),
                    stop=stop_after_attempt(20),
                    retry=retry_if_exception_type(ServerError)
            ):
                with attempt:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.0)
                    )
                    return response.text