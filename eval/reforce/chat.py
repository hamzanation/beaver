import sys
from abc import ABC, abstractmethod
from utils import extract_all_blocks
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from usage_utils import new_usage, add_usage

class BaseChat(ABC):
    def __init__(self, model: str, temperature: float = 1.0):
        self.model = model
        self.temperature = float(temperature)
        self.messages = []
        self.usage = new_usage()
        self.usage_latency_sec = 0.0
        self.usage_num_calls = 0
        self.usage_errors = 0

    def record_call_usage(self, usage, latency_sec, error=False):
        """Accumulate one LLM call's telemetry into this chat session's running totals."""
        add_usage(self.usage, usage)
        self.usage_latency_sec += latency_sec or 0.0
        self.usage_num_calls += 1
        if error:
            self.usage_errors += 1

    def get_token_usage(self):
        return {
            "usage": dict(self.usage),
            "latency_sec": self.usage_latency_sec,
            "num_calls": self.usage_num_calls,
            "errors": self.usage_errors,
        }

    @abstractmethod
    def get_response(self, prompt) -> str:
        pass

    def get_model_response(self, prompt, code_format=None) -> list:
        code_blocks = []
        max_try = 3
        while code_blocks == [] and max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
            code_blocks = extract_all_blocks(response, code_format)
        if max_try == 0 or code_blocks == []:
            print(f"get_model_response() exit, max_try: {max_try}, code_blocks: {code_blocks}")
            sys.exit(0)
        return code_blocks

    def get_model_response_txt(self, prompt) -> str:
        max_try = 3
        while max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
                return response
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
        print(f"get_model_response_txt() exit, max_try: {max_try}")
        sys.exit(0)

    def get_message_len(self):
        return {
            "prompt_len": sum(len(item["content"]) for item in self.messages if item["role"] == "user"),
            "response_len": sum(len(item["content"]) for item in self.messages if item["role"] == "assistant"),
            "num_calls": len(self.messages) // 2
        }

    def init_messages(self):
        self.messages = []


from openai import OpenAI, AzureOpenAI
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_llm import is_lmstudio_model, lmstudio_client, lmstudio_model_name, response_usage
from anthropic_llm import anthropic_chat_once, is_anthropic_model

class GPTChat(BaseChat):
    def __init__(self, azure=False, model="gpt-4o", temperature=1.0):
        super().__init__(model, temperature)
        self.is_lmstudio = False
        self.is_anthropic = False

        if not azure:
            if is_anthropic_model(model):
                self.is_anthropic = True
                self.client = None
            elif is_lmstudio_model(model):
                self.is_lmstudio = True
                self.model = lmstudio_model_name(model)
                self.client = lmstudio_client()
            elif model in ["o1-preview", "o1-mini"]:
                self.client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    api_version="2024-12-01-preview"
                )
            elif model in ["deepseek-reasoner"]:
                self.client = OpenAI(
                    base_url="https://api.deepseek.com",
                    api_key=os.environ.get("DS_API_KEY"),
                )
            elif "gemini" in model:
                self.client = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            elif "/" in model:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                )
            else:
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            if model in ["o1-preview", "o1-mini", "o3", "o4-mini"]:
                version = "2024-12-01-preview"
            elif model in ["o3-pro"]:
                version = "2025-03-01-preview"
            else:
                version = "2024-05-01-preview"

            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_KEY"),
                api_version=version
            )

    def get_response(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        usage = None
        if self.model == "o3-pro":
            response = self.client.responses.create(
                model=self.model,
                input=self.messages,
                temperature=self.temperature
            )
            main_content = response.output_text
            usage = response_usage(response)
        elif self.is_lmstudio:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=int(os.environ.get("BEAVER_MAX_TOKENS", "1024"))
            )
            main_content = response.choices[0].message.content
            usage = response_usage(response)
        elif self.is_anthropic:
            result = anthropic_chat_once(
                self.model,
                self.messages,
                temperature=self.temperature,
                max_tokens=int(os.environ.get("BEAVER_MAX_TOKENS", "1024")),
            )
            main_content = result["content"]
            usage = result["usage"]
        elif 'gemini' in self.model:
            if '2' in self.model:
                time.sleep(1)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature
                )
            else:
                response = self.client.chat.completions.create(
                model=self.model,
                reasoning_effort="none",
                messages=self.messages,
                temperature=self.temperature
            )
            main_content = response.choices[0].message.content
            usage = response_usage(response)
        elif 'claude' in self.model or 'qwen' in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                extra_body={
                    "reasoning": {
                        "effort": "none",
                    }
                },
            )
            main_content = response.choices[0].message.content
            usage = response_usage(response)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature
            )
            main_content = response.choices[0].message.content
            usage = response_usage(response)

        self.record_call_usage(usage, time.time() - t0)
        self.messages.append({"role": "assistant", "content": main_content})
        return main_content
