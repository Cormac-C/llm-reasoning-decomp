from transformers import AutoModelForCausalLM as Model
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import evaluate
import re


class ZebraPuzzleMetric(evaluate.Metric):
    def __init__(self):
        self.strict_accuracy = 0.0
        self.partial_accuracy = 0.0

    def _extract_and_split_parts(self, text):
        answer_starts = [
            "The solution is as follows:",
            "The solution is:",
            "Answer:",
            "Solution:",
            "Therefore,",
        ]

        # Find the beginning of the answer
        for start in answer_starts:
            if start.lower() in text.lower():
                return text.split(start)[1]
                break
        # Split the answer into parts
        split_regex = r"\n|,|\.|;"
        parts = re.split(split_regex, text)
        parts = [
            part.strip()
            for part in parts
            if part.strip()
            and part.strip().lower().startswith(("the", "here", "therefore"))
        ]
        return parts

    def _normalize_string(self, text):
        return re.sub(r"\s+", " ", text.lower().strip())

    def compute(self, predictions, references):
        strict_correct = 0
        partial_correct = 0
        num_examples = len(predictions)
        num_subparts = 0

        for pred, ref in zip(predictions, references):
            ref_parts = self._extract_and_split_parts(ref)
            pred_parts = self._extract_and_split_parts(pred)

            # Ignore first part of answer which is intro
            # TODO: Careful with this assumption
            ref_parts = ref_parts[1:]
            pred_parts = pred_parts[1:]

            correct_subparts = sum(
                any(
                    self._normalize_comparison(ref_part)
                    == self._normalize_comparison(pred_part)
                    for pred_part in pred_parts
                )
                for ref_part in ref_parts
            )

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


def compute_zebra_metrics(predictions, references):
    metric = ZebraPuzzleMetric()
    return metric.compute(predictions=predictions, references=references)


def eval_zebra(
    model: Model,
    eval_dataset: Dataset,
    tokenizer,
    formatting_prompts_func=None,
    response_template="#Answer",
    compute_metrics=compute_zebra_metrics,
):
    model.eval()

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )
    eval_dataset = eval_dataset.map(tokenizer, batched=True)
    trainer = SFTTrainer(
        model=model,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        compute_metrics=compute_zebra_metrics,
    )
    eval_metrics = trainer.evaluate()

    return eval_metrics
