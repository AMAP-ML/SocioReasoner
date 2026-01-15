import os
from typing import TYPE_CHECKING, Any, Dict, Union

import torch
import torch.nn as nn
from megatron.core.transformer.enums import AttnBackend

from ..constants import MCA_CONFIG_NAME
from ..utils import get_logger


if TYPE_CHECKING:
    from megatron.core.transformer import TransformerConfig


logger = get_logger(__name__)


class ModuleUtilsMixin:
    """
    inspired by HuggingFace Transformers
    """

    main_input_name: str = "input_ids"

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())
        total_numel = []
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                total_numel.append(param.numel())

        return sum(total_numel)

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)


class RMSNorm(nn.Module):
    def __init__(self, config: "TransformerConfig", hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        device = torch.cuda.current_device() if not config.use_cpu_initialization else None
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype, device=device))
        self.variance_epsilon = eps

        # set sequence parallelism flag
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def exists_hf_config(model_name_or_path: str) -> bool:
    return os.path.exists(os.path.join(model_name_or_path, "config.json"))


def exists_mca_config(model_name_or_path: str) -> bool:
    return os.path.exists(os.path.join(model_name_or_path, MCA_CONFIG_NAME))


def check_and_get_attention_backend_by_env(attention_backend: AttnBackend):
    if attention_backend != AttnBackend.auto:
        # user specified attention backend
        return attention_backend
    flash_attn = os.getenv("NVTE_FLASH_ATTN", None)
    fused_attn = os.getenv("NVTE_FUSED_ATTN", None)
    unfused_attn = os.getenv("NVTE_UNFUSED_ATTN", None)

    is_set_as = lambda env, value: env is not None and env == value

    if is_set_as(flash_attn, "0") and is_set_as(fused_attn, "0") and is_set_as(unfused_attn, "0"):
        return AttnBackend.local
    if is_set_as(flash_attn, "1") and (is_set_as(fused_attn, "0") or is_set_as(unfused_attn, "0")):
        return AttnBackend.flash
    if is_set_as(fused_attn, "1") and (is_set_as(flash_attn, "0") or is_set_as(unfused_attn, "0")):
        return AttnBackend.fused
    if is_set_as(unfused_attn, "1") and (is_set_as(flash_attn, "0") or is_set_as(fused_attn, "0")):
        return AttnBackend.unfused
    return AttnBackend.auto
