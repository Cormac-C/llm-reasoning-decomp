from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class LogiQA(Dataset):
    def __init__(self, max_len=512):
        self.dataset = load_dataset("lucasmccabe/logiqa")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-2-7b-hf")
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        context = item["context"]
        query = item["query"]
        options = item["options"]
        answer = item["correct_option"]

        question_with_context_options = context + " " + query + " " + " ".join(options)
        question_tokens = self.tokenizer(question_with_context_options, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")


        answer_tokens = self.tokenizer(answer, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")


        return {
            "input_ids": question_tokens["input_ids"].squeeze(),
            "attention_mask": question_tokens["attention_mask"].squeeze(),
            "labels": answer_tokens["input_ids"].squeeze(),
        }