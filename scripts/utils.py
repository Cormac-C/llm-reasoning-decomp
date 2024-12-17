import torch


def configure_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def read_int_arg(args, index, default=None):
    return int(args[index]) if index < len(args) else default


def clear_gpu_memory(model):
    model.zero_grad(set_to_none=True)

    model_device = next(model.parameters()).device
    model.to("cpu")

    torch.cuda.empty_cache()

    model.to(model_device)
    return model
