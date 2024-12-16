import os
import sys
import torch
import wandb

from peft import peft_model, PeftModel
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.utils import load_prep_zebra_dataset
from evals.zebra_eval import eval_model_zebra
from scripts.utils import configure_device

# Load environment variables
load_dotenv()

# Configure device
device = configure_device()

wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

wandb.init(project="Decomp")

ADAPTER_DIR = "/home/mila/x/xiaoyin.chen/scratch/projects/decomp/files/sos-1b/llama-instructsos-1b"

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


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
