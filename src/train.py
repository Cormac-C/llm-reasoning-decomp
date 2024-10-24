from peft import LoraConfig, get_peft_model
from transformers import Trainer, AutoModelForCausalLM as Model
from datasets import Dataset

# Default from: https://huggingface.co/blog/gemma-peft
default_lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)


# Train adapter function
def train_lora(
    base_model: Model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    adapter_name: str,
    lora_config: LoraConfig = default_lora_config,
    training_args=None,
    save_dir=None,
):
    peft_model = get_peft_model(
        model=base_model, peft_config=lora_config, adapter_name=adapter_name
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    if save_dir:
        peft_model.save_pretrained(save_dir, adapter_name)
