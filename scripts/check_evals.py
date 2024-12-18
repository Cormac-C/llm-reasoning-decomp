import os
import sys

from dotenv import load_dotenv
from transformers import AutoTokenizer

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.utils import load_prep_zebra_dataset
from evals.zebra_eval import (
    eval_model_zebra,
)
from scripts.utils import configure_device, read_named_args

# Load environment variables
load_dotenv()

# Configure device
device = configure_device()

args = read_named_args()

MODEL_NAME = args.base_model

FEW_SHOT = args.few_shot

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

zebra_dataset = load_prep_zebra_dataset(
    tokenizer, instruction_tuned=True, test_split_size=0.2, few_shot=FEW_SHOT
)

eval_dataset = zebra_dataset["test"]

metrics = eval_model_zebra(
    model=None,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print(metrics)
