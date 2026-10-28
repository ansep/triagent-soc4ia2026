from openai import OpenAI
from .settings import settings, Provider


def make_client(provider: Provider = None) -> OpenAI:
    """
    Returns an OpenAI-SDK client pointed at the right endpoint for each provider.
    Groq, DeepSeek, and llama.cpp all expose OpenAI-compatible /v1 endpoints,
    so we reuse the same SDK everywhere: no provider-specific imports needed.
    """
    p = provider or settings.backbone_provider

    if p == "groq":
        return OpenAI(
            api_key=settings.groq_api_key or "placeholder",
            base_url="https://api.groq.com/openai/v1",
        )
    if p == "deepseek":
        return OpenAI(
            api_key=settings.deepseek_api_key or "placeholder",
            base_url="https://api.deepseek.com/v1",
        )
    if p == "llamacpp":
        return OpenAI(
            api_key="lcpp",
            base_url=settings.llamacpp_base_url,
        )
    if p == "vllm":
        return OpenAI(api_key="vllm", base_url=settings.vllm_base_url)
    # default: openai
    return OpenAI(api_key=settings.openai_api_key or "placeholder")
