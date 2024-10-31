from datasets import load_dataset
import torch
from torch.utils.data import Dataset

def verbalize_solution(solution):
    verbalized_solution = "The solution is as follows:\n"
    headers = solution["header"]

    for row in solution["rows"]:
        row_description = f"In {headers[0].lower()} {row[0]}, "
        row_description += ", ".join(
            f"{headers[i].lower()} is {row[i]}" for i in range(1, len(headers))
        )
        row_description += ".\n"
        verbalized_solution += row_description

    return verbalized_solution

class Zebra(Dataset):
    def __init__(self, tokenizer, hf_token, split='test'):
        self.dataset = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", token=hf_token, split=split)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        puzzle_size = item["size"]
        puzzle = item["puzzle"]
        solution = item["solution"]

        # Format the question and verbalized answer
        question = f"Given {puzzle}. Please solve for the final arrangement."
        verbalized_answer = verbalize_solution(solution)

        input_text = f"{question} ### Answer: {verbalized_answer}"
        tokens = self.tokenizer(input_text, padding="longest", return_tensors="pt")

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
        }
