from transformers import AutoModelForCausalLM as Model, EvalPrediction
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
import numpy as np
from typing import Dict
import evaluate
import re

EPS = 1e-5
HF_MASK_VALUE = -100


class SudokuPuzzleMetric(evaluate.Metric):
    def __init__(self):
        self.strict_accuracy = 0.0
        self.partial_accuracy = 0.0
        self._puzzle_regex = re.compile(r"(\d{81})")

    def compute(self, predictions, references, num_clues_list):
        strict_correct = 0
        partial_correct = 0
        partial_correct_adjusted = 0
        num_examples = len(predictions)
        num_subparts = 0
        num_subparts_adjusted = 0

        for pred, ref, num_clues in zip(predictions, references, num_clues_list):
            ref_reg_search = self._puzzle_regex.search(ref)
            pred_reg_search = self._puzzle_regex.search(pred)

            ref_filtered = (
                ref_reg_search.group(1)
                if ref_reg_search
                else "".join([el for el in ref if el.isdigit()])
            )
            pred_filtered = (
                pred_reg_search.group(1)
                if pred_reg_search
                else "".join([el for el in pred if el.isdigit()])
            )

            # Split string by character
            ref_parts = list(ref_filtered)
            pred_parts = list(pred_filtered)

            correct_subparts = 0
            for ref_part, pred_part in zip(ref_parts, pred_parts):
                if ref_part == pred_part:
                    correct_subparts += 1

            correct_subparts_adjusted = correct_subparts - num_clues

            # Update totals
            num_subparts += len(ref_parts)
            num_subparts_adjusted += len(ref_parts) - num_clues

            if correct_subparts == len(ref_parts):
                strict_correct += 1
            partial_correct += correct_subparts
            partial_correct_adjusted += correct_subparts_adjusted

        self.strict_accuracy = strict_correct / (num_examples or EPS)
        self.partial_accuracy = partial_correct / (num_subparts or EPS)
        self.partial_accuracy_adjusted = partial_correct_adjusted / (
            num_subparts_adjusted or EPS
        )
        return {
            "strict_accuracy": self.strict_accuracy,
            "partial_accuracy": self.partial_accuracy,
            "partial_accuracy_adjusted": self.partial_accuracy_adjusted,
        }


def compute_sudoku_metrics(predictions, references, num_clues_list):
    metric = SudokuPuzzleMetric()
    metric_output = metric.compute(
        predictions=predictions, references=references, num_clues_list=num_clues_list
    )
    return {f"eval_{k}": v for k, v in metric_output.items()}


def generate_compute_metrics_fn(tokenizer, num_clues_list):
    def compute_sudoku_metrics_for_trainer(eval_preds: EvalPrediction) -> Dict:
        preds, labels = eval_preds

        preds = np.where(preds != HF_MASK_VALUE, preds, tokenizer.pad_token_id)
        labels = np.where(labels != HF_MASK_VALUE, labels, tokenizer.pad_token_id)

        preds_decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

        return compute_sudoku_metrics(preds_decoded, labels_decoded, num_clues_list)

    return compute_sudoku_metrics_for_trainer


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def eval_model_sudoku(
    model: Model,
    eval_dataset: Dataset,
    tokenizer,
    num_clues_list: list,
    formatting_prompts_func=None,
    response_template="<|start_header_id|>assistant<|end_header_id|>",
    content_key="formatted_text",
    save_dir="/tmp",
    run_name="",
):
    model.eval()

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )
    eval_dataset = eval_dataset.map(
        lambda examples: tokenizer(examples[content_key]), batched=True
    )

    training_args = SFTConfig(
        output_dir=save_dir,
        dataset_batch_size=1,
        report_to="wandb",
        run_name=run_name,
        eval_packing=False,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        label_names=["labels"],
    )

    trainer = SFTTrainer(
        model=model,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        compute_metrics=generate_compute_metrics_fn(tokenizer, num_clues_list),
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    eval_metrics = trainer.evaluate()

    return eval_metrics
