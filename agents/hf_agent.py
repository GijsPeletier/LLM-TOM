"""
hf_agent.py
Unified Agent for local Hugging Face models (Qwen, Llama, Mistral, etc.).
"""

import sys

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)
from agents.base_llm_agent import BaseLLMAgent


# Base system prompt instructing the model on the required JSON output format
SYSTEM_PROMPT = (
    "You are playing a negotiation game. Think step-by-step and show your full reasoning process. "
    "CRITICAL: You must enclose your final answer inside a JSON markdown block at the very end of your response. "
    "Do not use examples, but strictly follow this exact structure:\n\n"
    "```json\n"
    "{\n"
    '  "action": "<insert action string here>",\n'
    '  "offer": [<w>, <b>, <m>, <g>, <y>],\n'
    '  "message": "<insert short free-text message to opponent here>",\n'
    '  "reasoning": "<insert short strategy explanation here>"\n'
    "}\n"
    "```\n"
    "Ensure the keys 'action', 'offer', 'message', and 'reasoning' are strictly present, lowercase, and the offer is a list of exactly 5 integers."
)


class HFAgent(BaseLLMAgent):
    """Agent for running local Hugging Face models with quantization and shared memory caching."""

    _shared_models = {}
    _shared_tokenizers = {}

    def __init__(self, player_id, model_path, **kwargs):
        debug = kwargs.pop("debug", False)
        super().__init__(player_id, debug=debug)
        self.model_name = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Loads a Hugging Face model or retrieves it from cache. Applies appropriate quantization configurations."""
        if self.model is not None:
            return

        # Retrieve from cache if available
        if self.model_name in HFAgent._shared_models:
            print(f"[{self.player_id}] Using cached HF model: {self.model_name}...")
            self.model = HFAgent._shared_models[self.model_name]
            self.tokenizer = HFAgent._shared_tokenizers[self.model_name]
            return

        # Otherwise, load from scratch
        print(f"[{self.player_id}] Loading HF model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if "gpt-oss" in self.model_name.lower():
                # gpt-oss models ship with built-in Mxfp4 quantization.
                # Passing a BitsAndBytesConfig causes conflicts, so we load it as-is
                # and rely on the model's native configuration.
                print(
                    f"[{self.player_id}] gpt-oss detected: loading with built-in Mxfp4 quantization (no BnB config)..."
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            else:
                print(f"[{self.player_id}] Loading standard 8-bit model...")
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
                print(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )

            # Cache the model and tokenizer for future agents
            HFAgent._shared_models[self.model_name] = self.model
            HFAgent._shared_tokenizers[self.model_name] = self.tokenizer

        except Exception as e:
            print(f"[{self.player_id}] Failed to load {self.model_name}: {e}")
            raise

    def _generate_llm_response(self, prompt):
        """Formats the prompt, generates a response, and returns the newly generated tokens."""

        current_system = SYSTEM_PROMPT
        if "gpt-oss" in self.model_name.lower():
            current_system += "\nReasoning: high"

        messages = [
            {"role": "system", "content": current_system},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": 32768,
            "do_sample": False,
        }

        is_gpt_oss = "gpt-oss" in self.model_name.lower()
        if not is_gpt_oss:
            gen_kwargs["streamer"] = _TokenCountStreamer(
                tokenizer=self.tokenizer,
                player_id=f"P{self.player_id}",
                model_name=self.model_name,
            )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_length:]

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class _TokenCountStreamer(TextStreamer):
    def __init__(self, tokenizer, player_id, model_name):
        super().__init__(tokenizer, skip_special_tokens=True)
        self.player_id = player_id
        self.model_name = model_name
        self.token_count = 0
        self._printed_header = False

    def on_finalized_text(self, text, stream_end=False):
        if not self._printed_header:
            print(f"[HF] {self.player_id} streaming from {self.model_name}")
            self._printed_header = True
        if text:
            n = len(self.tokenizer(text, add_special_tokens=False).input_ids)
            self.token_count += n
            sys.stdout.write(f"\r   ↳ {self.token_count:>5d} tokens")
            sys.stdout.flush()
        if stream_end:
            sys.stdout.write("\n")
            sys.stdout.flush()
            print(f"[HF] {self.player_id} done — {self.token_count} tokens")
