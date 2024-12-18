import os
import sys
import wandb

from peft import peft_model, PeftModel
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.utils import load_prep_sudoku_dataset
from evals.sudoku_eval import eval_model_sudoku
from scripts.utils import configure_device, read_named_args, create_run_name

# Load environment variables
load_dotenv()

# Configure device
device = configure_device()

args = read_named_args(include_adapter_dir=True)

wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

wandb.init(project="Decomp", name=create_run_name(args, "sudoku-sos-eval"))

FEW_SHOT = args.few_shot

MODEL_NAME = args.base_model

ADAPTER_DIR = args.adapter_dir


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
dataset = load_prep_sudoku_dataset(
    tokenizer,
    instruction_tuned=True,
    few_shot=FEW_SHOT,
    test_split_size=0.2,
)

dataset = dataset["test"]

num_clues_list = dataset["num_clues"]

print(f"Loaded dataset: {len(dataset)} examples")

metrics = eval_model_sudoku(
    model=peft_model,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    num_clues_list=num_clues_list,
)

print(metrics)

wandb.log(metrics)
