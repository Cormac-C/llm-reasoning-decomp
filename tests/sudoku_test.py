from transformers import AutoModelForCausalLM as Model
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import numpy as np
import evaluate


class SudokuPuzzleMetric(evaluate.Metric):
    def __init__(self):
        self.strict_accuracy = 0.0
        self.partial_accuracy = 0.0

    def compute(self, predictions, references):
        strict_correct = 0
        partial_correct = 0
        num_examples = len(predictions)
        num_subparts = 0

        for pred, ref in zip(predictions, references):
            # Split string by character
            ref_parts = list(ref)
            pred_parts = list(pred)

            assert len(ref_parts) == len(pred_parts)
            assert len(ref_parts) == 81

            correct_subparts = 0
            for ref_part in ref_parts:
                if ref_part in pred_parts:
                    correct_subparts += 1

            # Update totals
            num_subparts += len(ref_parts)
            if correct_subparts == len(ref_parts):
                strict_correct += 1
            partial_correct += correct_subparts

        self.strict_accuracy = strict_correct / (num_examples or 1e-5)
        self.partial_accuracy = partial_correct / (num_subparts or 1e-5)
        return {
            "strict_accuracy": self.strict_accuracy,
            "partial_accuracy": self.partial_accuracy,
        }


def compute_sudoku_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metric = SudokuPuzzleMetric()
    metric_output = metric.compute(predictions=predictions, references=labels)
    return metric_output


def eval_baseline_sudoku(
    base_model: Model,
    eval_dataset: Dataset,
    tokenizer,
    formatting_prompts_func=None,
    response_template="#Answer",
    compute_metrics=compute_sudoku_metrics,
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
