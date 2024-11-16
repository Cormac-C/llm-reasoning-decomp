import os
import sys
import torch

from peft import LoraConfig
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.train import sft_train_lora
from src.model import identify_target_modules
from data.zebra import Zebra
from evals.zebra_eval import compute_zebra_metrics
from data.format import chat_format_qa_instance, lm_format_qa_instance

# Load environment variables
load_dotenv()

# Configure device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
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
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=os.environ["HF_TOKEN"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])

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

    sft_train_lora(
        base_model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        adapter_name="sft_lora",
        response_template="### Answer:",
        lora_config=lora_config,
        compute_metrics=compute_zebra_metrics,
        save_dir=save_dir,
    )

    pass