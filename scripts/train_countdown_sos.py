import os
import sys
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
from data.countdown import Countdown
from data.format import lm_format_qa_instance
from scripts.utils import configure_device, read_named_args, create_run_name


# Load environment variables
load_dotenv()

# Configure device
device = configure_device()

args = read_named_args()

wandb.login(key=os.environ["WANDB_KEY"], relogin=True, force=True)

RUN_NAME = create_run_name(args, base_name="countdown-training")

BASE_DIR = os.environ["BASE_DIR"]

MODEL_NAME = args.base_model

NUM_SAMPLES = args.num_samples


def get_sft_config(run_name=None):
    return SFTConfig(
        output_dir="/tmp",
        run_name=run_name,
        # Eval_strategy set to "no" temporarily cause of https://github.com/huggingface/transformers/issues/34701
        eval_strategy="no",
        eval_steps=100,
        eval_packing=False,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=1,
        report_to="wandb",
        logging_steps=10,
        dataset_batch_size=16,
        label_names=["labels"],
    )


def load_prep_countdown_sos(tokenizer, instruction_tuned=True, test_split_size=0.2):
    dataset = Countdown(json_file=os.environ["COUNTDOWN_PATH"], num_samples=NUM_SAMPLES)

    few_shot_example = dataset[0]
    few_shot_prompt = (
        f"Here is an example:\n"
        f"{few_shot_example['question']}"
        f"{few_shot_example['answer']}"
    )

    if instruction_tuned:
        formatted_list = [
            format_with_few_shot(example, few_shot_prompt, instruction_tuned)
            for example in dataset
        ]
        formatted_list = tokenizer.apply_chat_template(
            formatted_list, tokenize=False, add_generation_prompt=False
        )
    else:
        formatted_list = [lm_format_qa_instance(example) for example in dataset]

    dataset = Dataset.from_dict({"formatted_text": formatted_list})
    dataset = dataset.train_test_split(test_size=test_split_size)
    return dataset


def format_with_few_shot(example, few_shot_prompt, use_chat_format=True):
    task_description = (
        "You are tasked to solve arithmetic reasoning problems. "
        "Given a set of numbers and a target, describe the steps in the path to reach the target using those numbers."
    )
    guidelines = (
        "Using arithmetic operations such as addition (+), subtraction (-), multiplication (*) and division (/), "
        "use the initial set of numbers to gather new numbers that eventually reach the target in the end."
    )

    # Format the dataset using the appropriate format
    if use_chat_format:
        return [
            {
                "role": "user",
                "content": f"{task_description}\n{guidelines}\n{few_shot_prompt}\n{example["question"]}",
            },
            {"role": "assistant", "content": example["answer"]},
        ]

    else:
        return (
            f"### Question {task_description}\n{guidelines}\n{few_shot_prompt}\n{example["question"]}\n"
            f"### Answer {example["answer"]}"
        )


def train_countdown_sos(
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

    dataset = load_prep_countdown_sos(
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
        ),
        dataset,
    )


save_dir = BASE_DIR + RUN_NAME


tokenizer, trained_model, dataset = train_countdown_sos(
    instruction_tuned=True,
    model_name=MODEL_NAME,
    test_split_size=0.2,
    save_dir=save_dir,
    run_name=RUN_NAME,
)
