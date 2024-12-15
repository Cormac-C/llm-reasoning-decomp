import os
import sys
import torch
import wandb

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from evals.sudoku_eval import eval_model_sudoku
from data.utils import load_prep_sudoku_dataset

# Load environment variables
load_dotenv()

# Configure device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

wandb.init(project="Decomp", name="base-sudoku-1b-zero-shot")

FEW_SHOT = None

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=os.environ["HF_TOKEN"],
    torch_dtype="auto",
    device_map="auto",
)

model.to(device)

print(f"Loaded model: {MODEL_NAME}")
print(f"Model precision: {model.config.torch_dtype}")

model.eval()

# Load dataset
dataset = load_prep_sudoku_dataset(
    tokenizer, instruction_tuned=True, test_split_size=0.2, few_shot=FEW_SHOT
)

dataset = dataset["test"]

num_clues_list = dataset["num_clues"]

print(f"Loaded dataset: {len(dataset)} examples")

metrics = eval_model_sudoku(
    model=model,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    num_clues_list=num_clues_list,
)

print(metrics)

wandb.log(metrics)
