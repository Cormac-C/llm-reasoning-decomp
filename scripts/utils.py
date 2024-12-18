import torch
import argparse


def configure_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def read_named_args(include_adapter_dir=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot", type=int, default=None)
    parser.add_argument("--base_model", type=str, default="1B")
    if include_adapter_dir:
        parser.add_argument("adapter_dir", type=str, default=None)

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


def create_run_name(args, base_name=""):
    run_name = base_name

    if args.few_shot is not None:
        run_name += f"-{args.few_shot}shot"
    else:
        run_name += "-zeroshot"

    if args.base_model == "meta-llama/Llama-3.2-1B-Instruct":
        run_name += "-1B"
    elif args.base_model == "meta-llama/Llama-3.2-3B-Instruct":
        run_name += "-3B"
    else:
        run_name += args.base_model
    
    if args.num_samples is not None:
        run_name += f"-{args.num_samples}samples"
        
    return run_name


def clear_gpu_memory(model):
    model.zero_grad(set_to_none=True)

    model_device = next(model.parameters()).device
    model.to("cpu")

    torch.cuda.empty_cache()

    model.to(model_device)
    return model
