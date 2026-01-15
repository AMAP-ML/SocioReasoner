import vllm

LLM = None
AsyncLLM = None

if "0.7.3" in vllm.__version__:
    from roll.third_party.vllm.vllm_0_7_3.llm import Llm073
    LLM = Llm073
elif "0.8.4" in vllm.__version__:
    from roll.third_party.vllm.vllm_0_8_4.llm import Llm084
    from roll.third_party.vllm.vllm_0_8_4.v1.async_llm import AsyncLLM084
    LLM = Llm084
    AsyncLLM = AsyncLLM084
else:
    raise NotImplementedError(f"roll vllm version {vllm.__version__} is not supported.")

__all__ = ["LLM", "AsyncLLM"]
