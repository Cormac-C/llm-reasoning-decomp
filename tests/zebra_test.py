from transformers import AutoModelForCausalLM as Model
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import numpy as np
import evaluate


def compute_zebra_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # Calculate the model accuracy
    # TODO: Update metrics to be puzzle specific
    metric = evaluate.load("accuracy")
    metric_output = metric.compute(predictions=predictions, references=labels)
    return metric_output


def eval_baseline_zebra(
    base_model: Model,
    eval_dataset: Dataset,
    tokenizer,
    formatting_prompts_func=None,
    response_template="#Answer",
    compute_metrics=compute_zebra_metrics,
):
    base_model.eval()

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )
    eval_dataset = eval_dataset.map(
        lambda examples: tokenizer(examples["input_text"]), batched=True
    )
    trainer = SFTTrainer(
        model=base_model,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    eval_metrics = trainer.evaluate()

    return eval_metrics
