import os
import sys
import wandb

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from evals.zebra_eval import eval_model_zebra
from data.utils import load_prep_zebra_dataset
from scripts.utils import configure_device, read_named_args

# Load environment variables
load_dotenv()

# Configure device
device = configure_device()

args = read_named_args()

wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

# TODO: Create run name based on args
wandb.init(project="Decomp", name="base-zebra-3b-zero-shot")

FEW_SHOT = args.few_shot

MODEL_NAME = args.base_model

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
dataset = load_prep_zebra_dataset(
    tokenizer, instruction_tuned=True, test_split_size=0.2, few_shot=FEW_SHOT
)

dataset = dataset["test"]

print(f"Loaded dataset: {len(dataset)} examples")

metrics = eval_model_zebra(model=model, eval_dataset=dataset, tokenizer=tokenizer)

print(metrics)

wandb.log(metrics)
