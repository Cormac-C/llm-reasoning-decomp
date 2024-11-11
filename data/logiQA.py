from datasets import load_dataset
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

        question_with_context_options = (
            f"{context} {query} {' '.join(options)}"
        )

        return {"question": question_with_context_options, "answer": answer}
