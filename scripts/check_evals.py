import os
import sys
import torch

from dotenv import load_dotenv
from transformers import AutoTokenizer

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.utils import load_prep_zebra_dataset
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
