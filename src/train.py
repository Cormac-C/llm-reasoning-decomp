from peft import LoraConfig, get_peft_model
from transformers import Trainer, AutoModelForCausalLM as Model
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


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


default_sft_config = SFTConfig(
    output_dir="/tmp",
    eval_strategy="epoch",
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


def sft_train_lora(
    base_model: Model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    adapter_name: str,
    formatting_prompts_func=None,
    response_template="#Answer",
    lora_config: LoraConfig = default_lora_config,
    training_args=default_sft_config,
    compute_metrics=None,
    save_dir="/tmp",
):
    peft_model = get_peft_model(
        model=base_model, peft_config=lora_config, adapter_name=adapter_name
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    train_dataset = train_dataset.map(
        lambda examples: tokenizer(examples["input_text"]), batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda examples: tokenizer(examples["input_text"]), batched=True
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if save_dir:
        peft_model.save_pretrained(save_dir, adapter_name)
