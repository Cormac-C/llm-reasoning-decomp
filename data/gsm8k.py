from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class GSM8K(Dataset):
    def __init__(self, max_len=512):
        self.dataset = load_dataset("openai/gsm8k", "main")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-2-7b-hf")
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer = item["answer"]

        question_tokens = self.tokenizer(question, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        answer_tokens = self.tokenizer(answer, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": question_tokens["input_ids"].squeeze(),
            "attention_mask": question_tokens["attention_mask"].squeeze(),
            "labels": answer_tokens["input_ids"].squeeze(),
        }
