from transformers import AutoModelForCausalLM as Model
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from torch.utils.data import DataLoader
import torch
import wandb
import evaluate
import re


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

    def _grade_part_pair(self, ref_parts, pred_parts):
        num_correct = 0
        ref_parts_split = ref_parts.split(",")[1:]  # Skip the house number
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

            print(f"Partial correct: {iter_partial_correct}/{iter_subparts}")
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
    device="cuda",
):
    model.eval()

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    eval_dataset = eval_dataset.map(
        lambda examples: tokenizer(examples[content_key]), batched=True
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=1, collate_fn=collator, shuffle=False
    )

    # Initialize predictions and references lists
    all_preds = []
    all_refs = []

    # Evaluation loop
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to device
            inputs = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            # Generate predictions
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
            )

            # Decode predictions
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(decoded_preds)

            # Store references
            decoded_refs = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            all_refs.extend(decoded_refs)

            # Clear GPU memory
            torch.cuda.empty_cache()

    # Compute metrics
    eval_metrics = compute_metrics(predictions=all_preds, references=all_refs)

    # Log to wandb
    wandb.log(eval_metrics)

    # training_args = SFTConfig(
    #     output_dir=save_dir,
    #     dataset_batch_size=1,
    #     report_to="wandb",
    #     eval_accumulation_steps=1,
    # )

    # trainer = SFTTrainer(
    #     model=model,
    #     eval_dataset=eval_dataset,
    #     formatting_func=formatting_prompts_func,
    #     data_collator=collator,
    #     compute_metrics=compute_metrics,
    #     args=training_args,
    # )
    # eval_metrics = trainer.evaluate()
    return eval_metrics
