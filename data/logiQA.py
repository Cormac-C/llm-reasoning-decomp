from datasets import load_dataset
import torch
from torch.utils.data import Dataset

class LogiQA(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.dataset = load_dataset("lucasmccabe/logiqa", split=split)
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        context = item["context"]
        query = item["query"]
        options = item["options"]
        answer = item["correct_option"]
        print(f"{context} {query} {' '.join(options)} ### Answer: {answer}")

        question_with_context_options = f"{context} {query} {' '.join(options)} ### Answer: {answer}"
        tokens = self.tokenizer(question_with_context_options, padding="longest", return_tensors="pt")

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
        }
