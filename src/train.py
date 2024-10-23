from peft import LoraConfig
from transformers import Trainer
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
    base_model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    adapter_name: str,
    lora_config: LoraConfig = default_lora_config,
    training_args=None,
    save_dir=None,
):
    base_model.add_adapter(lora_config, adapter_name)

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    if save_dir:
        base_model.save_pretrained(save_dir, adapter_name)
