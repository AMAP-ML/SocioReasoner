# borrow from https://github.com/volcengine/verl/blob/main/verl/utils/vllm_utils.py

SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM
    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)):
        return

    for layer in model.model.layers:
        mlp = getattr(layer, "mlp")
        param_dict = dict(mlp.named_parameters())
        for name, param in param_dict.items():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = mlp.experts.weight_loader
