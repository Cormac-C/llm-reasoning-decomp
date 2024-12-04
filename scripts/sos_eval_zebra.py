import os
import sys
import torch
import wandb

from peft import peft_model, PeftModel
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import Subset

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.zebra import Zebra
from data.format import (
    chat_format_qa_instance,
    lm_format_qa_instance,
    chat_create_fewshot_prompt,
)
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

wandb.init(project="Decomp")

ADAPTER_DIR = "/home/mila/x/xiaoyin.chen/scratch/projects/decomp/files/sos-3b/llama-instructsos-3b"

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def load_prep_zebra_dataset(
    tokenizer, instruction_tuned=True, few_shot=None, test_split_size=0.2
):
    dataset = Zebra(hf_token=os.environ["HF_TOKEN"])
    if instruction_tuned:
        formatted_list = []
        if few_shot is not None:
            fewshot_examples = Subset(dataset, range(few_shot))
            dataset = Subset(dataset, range(few_shot, len(dataset)))
            formatted_list = [
                chat_create_fewshot_prompt(
                    example, examples=fewshot_examples, num_shots=few_shot
                )
                for example in dataset
            ]
        else:
            formatted_list = [chat_format_qa_instance(example) for example in dataset]
        formatted_list = tokenizer.apply_chat_template(
            formatted_list, tokenize=False, add_generation_prompt=False
        )
    else:
        formatted_list = [lm_format_qa_instance(example) for example in dataset]
    dataset = Dataset.from_dict({"formatted_text": formatted_list})

    dataset = dataset.train_test_split(test_size=test_split_size)
    return dataset


# Load base model and adapter
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"],
    torch_dtype="auto",
    device_map="auto",
)

peft_model = PeftModel.from_pretrained(model, ADAPTER_DIR, "sos")
peft_model.to(device)

print(f"Loaded model: {MODEL_NAME}")
print(f"Model precision: {model.config.torch_dtype}")

peft_model.eval()

# Load dataset
dataset = load_prep_zebra_dataset(
    tokenizer, instruction_tuned=True, test_split_size=0.2, few_shot=None
)

dataset = dataset["test"]

print(f"Loaded dataset: {len(dataset)} examples")

metrics = eval_model_zebra(model=peft_model, eval_dataset=dataset, tokenizer=tokenizer)

print(metrics)

wandb.log(metrics)
