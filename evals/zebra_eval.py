from transformers import AutoModelForCausalLM as Model
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
import evaluate
import re


class ZebraPuzzleMetric(evaluate.Metric):
    def __init__(self):
        self.strict_accuracy = 0.0
        self.partial_accuracy = 0.0

    def _extract_and_split_parts(self, text):
        # Each part should correspond to one house in the zebra puzzle
        answer_starts = [
            "The solution is as follows:",
            "The solution is:",
            "Answer:",
            "Solution:",
            "Therefore,",
            "as follows:",
        ]
        answer_starts_re = re.compile(
            "|".join(re.escape(start) for start in answer_starts),
            flags=re.IGNORECASE,
        )

        match = answer_starts_re.search(text)
        if match:
            text = text[match.end() :]

        # Split the answer into parts (should correspond to the number of sentences)
        split_regex = r"\n|\.|•|;"
        parts = re.split(split_regex, text)
        parts = [part.strip() for part in parts if part.strip()]
        return parts

    def _normalize_string(self, text):
        return re.sub(r"\s+", " ", text.lower().strip())

    def _extract_house_number(self, text):
        house_pattern = re.compile(r"House (\d+)", re.IGNORECASE)
        house_num = house_pattern.search(text)
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

    def _grade_part_pair(self, ref_parts, pred_parts):
        num_correct = 0
        ref_parts_split = ref_parts.split(",")
        ref_parts_split = ref_parts_split[1:]  # Skip the house number
        pred_parts_split = pred_parts.split(",")
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

        self.strict_accuracy = strict_correct / (num_examples or 1e-5)
        self.partial_accuracy = partial_correct / (num_subparts or 1e-5)
        return {
            "strict_accuracy": self.strict_accuracy,
            "partial_accuracy": self.partial_accuracy,
        }


def compute_zebra_metrics(predictions, references):
    metric = ZebraPuzzleMetric()
    return metric.compute(predictions=predictions, references=references)


def eval_model_zebra(
    model: Model,
    eval_dataset: Dataset,
    tokenizer,
    formatting_prompts_func=None,
    response_template="<|start_header_id|>assistant<|end_header_id|>",
    compute_metrics=compute_zebra_metrics,
    content_key="formatted_text",
    save_dir="/tmp",
):
    model.eval()

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    eval_dataset = eval_dataset.map(
        lambda examples: tokenizer(examples[content_key]), batched=True
    )

    training_args = SFTConfig(
        output_dir=save_dir, eval_accumulation_steps=20, report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args,
    )
    eval_metrics = trainer.evaluate()
    return eval_metrics
