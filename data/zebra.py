from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class Zebra(Dataset):
    def __init__(self, max_len=512):
        self.dataset = load_dataset("allenai/ZebraLogicBench-private", "grid_mode")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-2-7b-hf")
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        puzzle_size = item["size"]
        puzzle = item["puzzle"]
        solution = item["solution"]

        question = f"Given a {puzzle_size} grid, {puzzle}. What is the solution?"
        question_tokens = self.tokenizer(question, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        answer_tokens = self.tokenizer(solution, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": question_tokens["input_ids"].squeeze(),
            "attention_mask": question_tokens["attention_mask"].squeeze(),
            "labels": answer_tokens["input_ids"].squeeze(),
        }