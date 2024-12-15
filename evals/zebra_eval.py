from transformers import AutoModelForCausalLM as Model, EvalPrediction
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
import evaluate
import re
from typing import Dict
import numpy as np

EPS = 1e-5
HF_MASK_VALUE = -100


class ZebraPuzzleMetric(evaluate.Metric):
    def __init__(self):
        self.strict_accuracy = 0.0
        self.partial_accuracy = 0.0

        self._answer_starts = [
            "The solution is as follows:",
            "The solution is:",
            "Answer:",
            "Solution:",
            "Therefore,",
            "as follows:",
        ]
        self._answer_start_pattern = re.compile(
            "|".join(re.escape(start) for start in self._answer_starts),
            flags=re.IGNORECASE,
        )
        self._split_pattern = re.compile(r"\n|\.|•|;")
        self._whitespace_pattern = re.compile(r"\s+")
        self._house_pattern = re.compile(r"House (\d+)", re.IGNORECASE)

    def _extract_and_split_parts(self, text):
        # Each part should correspond to one house in the zebra puzzle
        match = self._answer_start_pattern.search(text)
        if match:
            text = text[match.end() :]

        # Split the answer into parts (should correspond to the number of sentences)
        parts = [
            part.strip() for part in self._split_pattern.split(text) if part.strip()
        ]
        return parts

    def _normalize_string(self, text):
        return self._whitespace_pattern.sub(" ", text.lower().strip())

    def _extract_house_number(self, text):
        house_num = self._house_pattern.search(text)
        house_num = int(house_num.group(1)) if house_num else None
        return house_num

    def _match_parts(self, ref_parts, pred_parts):
        # Match each part of the reference with a part of the prediction based on house number
        part_pairs = {}
        for ref_part in ref_parts:
            # Find house number in reference part
            house_num = self._extract_house_number(ref_part)
            if house_num is None:
                continue
            else:
                part_pairs[house_num] = {"ref": ref_part, "pred": None}
        for pred_part in pred_parts:
            # Find house number in prediction part
            house_num = self._extract_house_number(pred_part)
            if house_num is None:
                continue
            else:
                if house_num in part_pairs:
                    part_pairs[house_num]["pred"] = pred_part
        return part_pairs

    def _grade_part_pair(self, ref_parts="", pred_parts=""):
        num_correct = 0
        ref_parts_split = (
            ref_parts.split(",")[1:] if ref_parts is not None else []
        )  # Skip the house number
        pred_parts_split = pred_parts.split(",") if pred_parts is not None else []
        for ref_part in ref_parts_split:
            detail = ref_part.split("is ")[1]
            for pred_part in pred_parts_split:
                if detail in pred_part:
                    num_correct += 1
                    break
        return num_correct, len(ref_parts_split)

    def compute(self, predictions, references):
        strict_correct = 0
        partial_correct = 0
        num_examples = len(predictions)
        num_subparts = 0

        for pred, ref in zip(predictions, references):
            iter_subparts = 0
            iter_partial_correct = 0

            ref_parts = self._extract_and_split_parts(ref)
            pred_parts = self._extract_and_split_parts(pred)

            # Match each part of the reference with a part of the prediction based on house number
            part_pairs = self._match_parts(ref_parts, pred_parts)
            print(f"part pairs: {part_pairs}")

            for house_num, parts in part_pairs.items():
                ref_parts = parts["ref"]
                pred_parts = parts["pred"]
                correct_subparts, total_subparts = self._grade_part_pair(
                    ref_parts, pred_parts
                )
                iter_partial_correct += correct_subparts
                iter_subparts += total_subparts

            # Update totals
            num_subparts += iter_subparts
            if iter_subparts == iter_partial_correct:
                strict_correct += 1
            partial_correct += iter_partial_correct

        self.strict_accuracy = strict_correct / (num_examples or EPS)
        self.partial_accuracy = partial_correct / (num_subparts or EPS)
        return {
            "strict_accuracy": self.strict_accuracy,
            "partial_accuracy": self.partial_accuracy,
        }


def compute_zebra_metrics(predictions, references):
    metric = ZebraPuzzleMetric()
    results = metric.compute(predictions=predictions, references=references)
    return {f"eval_{k}": v for k, v in results.items()}


def generate_compute_metrics_fn(tokenizer):
    def compute_zebra_metrics_for_trainer(eval_preds: EvalPrediction) -> Dict:
        preds, labels = eval_preds

        preds = np.where(preds != HF_MASK_VALUE, preds, tokenizer.pad_token_id)
        labels = np.where(labels != HF_MASK_VALUE, labels, tokenizer.pad_token_id)

        preds_decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

        return compute_zebra_metrics(preds_decoded, labels_decoded)

    return compute_zebra_metrics_for_trainer


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def eval_model_zebra(
    model: Model,
    eval_dataset: Dataset,
    tokenizer,
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
        lambda examples: tokenizer(examples[content_key]),
        batched=True,
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
        compute_metrics=generate_compute_metrics_fn(tokenizer),
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    eval_metrics = trainer.evaluate()

    return eval_metrics
