import os
import sys
import torch
import wandb

from peft import LoraConfig
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTConfig

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.train import sft_train_lora
from src.model import identify_target_modules
from data.zebra import Zebra
from data.format import chat_format_qa_instance, lm_format_qa_instance
from evals.zebra_eval import eval_model_zebra

# Load environment variables
load_dotenv()

# Configure device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

RUN_NAME = "zebra-3b"

BASE_DIR = "/home/mila/x/xiaoyin.chen/scratch/projects/decomp/files/"

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def clear_gpu_memory(model):
    model.zero_grad(set_to_none=True)

    model_device = next(model.parameters()).device
    model.to("cpu")

    torch.cuda.empty_cache()

    model.to(model_device)
    return model


def log_zebra_metrics(metrics, run_name):
    wandb.log(
        {
            "zebra/strict_accuracy": metrics["strict_accuracy"],
            "zebra/partial_accuracy": metrics["partial_accuracy"],
        },
        run=run_name,
    )


def get_sft_config(run_name=None):
    return SFTConfig(
        output_dir="/tmp",
        run_name=run_name,
        # Eval_strategy set to "no" temporarily cause of https://github.com/huggingface/transformers/issues/34701
        eval_strategy="no",
        report_to="wandb",
        logging_steps=10,
        dataset_batch_size=16,
    )


def load_prep_zebra_dataset(tokenizer, instruction_tuned=True, test_split_size=0.2):
    dataset = Zebra(hf_token=os.environ["HF_TOKEN"])
    if instruction_tuned:
        formatted_list = [chat_format_qa_instance(example) for example in dataset]
        formatted_list = tokenizer.apply_chat_template(
            formatted_list, tokenize=False, add_generation_prompt=False
        )
    else:
        formatted_list = [lm_format_qa_instance(example) for example in dataset]
    dataset = Dataset.from_dict({"formatted_text": formatted_list})

    dataset = dataset.train_test_split(test_size=test_split_size)
    return dataset


def train_zebra_baseline(
    instruction_tuned=True,
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    test_split_size=0.2,
    save_dir="/tmp",
    run_name=None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ["HF_TOKEN"],
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"Loaded model: {model_name}")
    print(f"Model precision: {model.config.torch_dtype}")

    dataset = load_prep_zebra_dataset(
        tokenizer=tokenizer,
        instruction_tuned=instruction_tuned,
        test_split_size=test_split_size,
    )

    lora_config = LoraConfig(
        target_modules=identify_target_modules(model, name_segment="self_attn"),
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )

    training_config = get_sft_config(run_name=run_name)

    adapter_name = "llama-instruct" + run_name

    return (
        tokenizer,
        sft_train_lora(
            base_model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            adapter_name=adapter_name,
            response_template="<|start_header_id|>assistant<|end_header_id|>",
            lora_config=lora_config,
            training_args=training_config,
            save_dir=save_dir,
        ),
        dataset,
    )


save_dir = BASE_DIR + RUN_NAME

tokenizer, trained_model, dataset = train_zebra_baseline(
    instruction_tuned=True,
    model_name=MODEL_NAME,
    test_split_size=0.15,
    save_dir=save_dir,
    run_name=RUN_NAME,
)

# Clear GPU cache except for model and dataset
print("Clearing GPU memory before eval")
clear_gpu_memory(trained_model)


# Evaluate the trained model
metrics = eval_model_zebra(
    model=trained_model,
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    save_dir=save_dir,
    run_name=RUN_NAME + "-eval",
)
wandb.log(metrics)
log_zebra_metrics(metrics, run_name=RUN_NAME)
