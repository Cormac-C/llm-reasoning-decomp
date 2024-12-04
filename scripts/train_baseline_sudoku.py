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
from data.sudoku import Sudoku
from data.format import chat_format_qa_instance, lm_format_qa_instance
from evals.sudoku_eval import (
    eval_model_sudoku,
    generate_compute_metrics_fn,
    preprocess_logits_for_metrics
)

# Load environment variables
load_dotenv()

# Configure device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backend.mps.is_available() else "cpu"
)

wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

RUN_NAME = "sudoku-1b"

BASE_DIR = "/home/mila/x/xiaoyin.chen/scratch/projects/decomp/files/"

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

def clear_gpu_memory(model):
    model.zero_grad(set_to_none=True)

    model_device = next(model.parameters()).device
    model.to("cpu")

    torch.cuda.empty_cache()

    model.to(model_device)
    return model

def get_sft_config(run_name=None):
    return SFTConfig(
        output_dir="/tmp",
        run_name=run_name,
        # Eval_strategy set to "no" temporarily cause of https://github.com/huggingface/transformers/issues/34701
        eval_strategy="steps",
        eval_steps=100,
        eval_packing=False,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=1,
        report_to="wandb",
        logging_steps=10,
        dataset_batch_size=16,
        label_names=["labels"],
    )


def load_prep_sudoku_dataset(tokenizer, instruction_tuned=True, test_split_size=0.2):
    dataset = Sudoku(data_file=os.environ["SUDOKU_PATH"])

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


def train_sudoku_baseline(
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
    
    dataset = load_prep_sudoku_dataset(
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
            compute_metrics=generate_compute_metrics_fn(tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        ),
        dataset,
    )


save_dir = BASE_DIR + RUN_NAME

try:
    tokenizer, trained_model, dataset = train_sudoku_baseline(
        instruction_tuned=True,
        model_name=MODEL_NAME,
        test_split_size=0.2,
        save_dir=save_dir,
        run_name=RUN_NAME
    )
except Exception as e:
    print(f"Encountered exception: {e}")
    pass

clear_gpu_memory(trained_model)

metrics = eval_model_sudoku(
    model=trained_model, eval_dataset=dataset["test"], tokenizer=tokenizer
)

wandb.log(metrics)