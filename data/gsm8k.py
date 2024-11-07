from datasets import load_dataset
from torch.utils.data import Dataset


class GSM8K(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.dataset = load_dataset("openai/gsm8k", "main", split=split)
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer = item["answer"]

        input_text = f"{question} ### Answer: {answer}"

        return {"input_text": input_text}
