from torch import nn
from transformers import pytorch_utils as torch_utils


def identify_target_modules(
    model,
    name_segment="attn",
    module_types=(nn.Linear, nn.Embedding, nn.Conv2d, torch_utils.Conv1D),
):
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            if name_segment in name:
                target_modules.append(name)
