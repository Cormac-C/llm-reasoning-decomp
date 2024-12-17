import torch
import argparse


def configure_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def read_named_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot", type=int, default=None)
    parser.add_argument("--base_model", type=str, default="1B")
    parsed = parser.parse_args()

    if parsed.few_shot is not None and parsed.few_shot == 0:
        parsed.few_shot = None

    if parsed.base_model == "1B":
        parsed.base_model = "meta-llama/Llama-3.2-1B-Instruct"
    elif parsed.base_model == "3B":
        parsed.base_model = "meta-llama/Llama-3.2-3B-Instruct"
    else:
        raise ValueError("Invalid base model")

    return parsed


def clear_gpu_memory(model):
    model.zero_grad(set_to_none=True)

    model_device = next(model.parameters()).device
    model.to("cpu")

    torch.cuda.empty_cache()

    model.to(model_device)
    return model
