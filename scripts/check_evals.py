import os
import sys
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import Dataset

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.zebra import Zebra
from data.format import chat_format_qa_instance, lm_format_qa_instance
from evals.zebra_eval import (
    eval_model_zebra_no_trainer,
)

# Load environment variables
load_dotenv()

# Configure device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


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


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

zebra_dataset = load_prep_zebra_dataset(
    tokenizer, instruction_tuned=True, test_split_size=0.2
)

eval_dataset = zebra_dataset["test"]

metrics = eval_model_zebra_no_trainer(
    model=None,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print(metrics)
