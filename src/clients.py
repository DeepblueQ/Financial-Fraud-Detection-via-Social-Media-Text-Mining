# src/llm_client.py
import os
from openai import OpenAI


def _get_api_key(env_name: str) -> str:
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(f"lack环境变量 {env_name} ")
    return key


def build_client(provider: str):
    if provider == "openai":
        return OpenAI(api_key=_get_api_key("OPENAI_API_KEY"))

    if provider == "qwen":
        return OpenAI(
            api_key=_get_api_key("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    if provider == "deepseek":
        return OpenAI(
            api_key=_get_api_key("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

    raise ValueError(f"Unknown provider: {provider}")
