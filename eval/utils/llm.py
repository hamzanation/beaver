import re
from abc import ABC, abstractmethod
from openai import OpenAI
import os
import json
import time

from local_llm import is_lmstudio_model, lmstudio_client, lmstudio_model_name, response_usage
from anthropic_llm import anthropic_chat_once, is_anthropic_model
from usage_utils import new_usage, add_usage

def extract_json_block(text):
    """
    Extracts a JSON block from the text. 
    Tries to find ```json ... ``` or just {...} 
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match: return match.group(1)
    return text

class BaseChat(ABC):
    def __init__(self, model: str, temperature: float = 1.0):
        self.model = model
        self.temperature = float(temperature)
        self.messages = []
        self.usage = new_usage()
        self.usage_latency_sec = 0.0
        self.usage_num_calls = 0
        self.usage_errors = 0

    @abstractmethod
    def get_response(self, prompt) -> str:
        pass

    def record_call_usage(self, usage, latency_sec, error=False):
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

class GPTChat(BaseChat):
    def __init__(self, model="gpt-5-mini"):
        super().__init__(model)
        self.is_lmstudio = is_lmstudio_model(model)
        self.is_anthropic = is_anthropic_model(model)
        if self.is_anthropic:
            self.client = None
        elif self.is_lmstudio:
            self.model = lmstudio_model_name(model)
            self.client = lmstudio_client()
        else:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_response(self, prompt, json_mode=False) -> str:
        self.messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            kwargs = {
                "model": self.model,
                "messages": self.messages,
            }
            if self.is_anthropic:
                result = anthropic_chat_once(
                    self.model,
                    self.messages,
                    temperature=0,
                    max_tokens=int(os.environ.get("BEAVER_MAX_TOKENS", "1024")),
                )
                self.record_call_usage(result["usage"], time.time() - t0)
                return result["content"]
            if json_mode:
                # "json_object" requires the model to support it (e.g. gpt-4-1106-preview or later)
                # and usually requires the word "JSON" in the prompt (which we have).
                if not self.is_lmstudio:
                    kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            self.record_call_usage(response_usage(response), time.time() - t0)
            return response.choices[0].message.content
        except Exception as e:
            self.record_call_usage(None, time.time() - t0, error=True)
            print(f"Error calling LLM: {e}")
            raise e

    def get_json_response(self, prompt, retries=3):
        for attempt in range(retries):
            try:
                response_text = self.get_response(prompt, json_mode=True)
                json_str = extract_json_block(response_text)
                return json.loads(json_str)
            except json.JSONDecodeError:
                print(f"JSON decode error on attempt {attempt+1}. Retrying...")
                continue
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {e}")
                time.sleep(2)
                continue
        return None
