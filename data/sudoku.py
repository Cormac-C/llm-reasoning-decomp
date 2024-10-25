import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class Sudoku(Dataset):
    def __init__(self, data_file, max_len=81):
        self.dataset = pd.read_csv(data_file)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-2-7b-hf")
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        puzzle = item["puzzle"]
        solution = item["solution"]
        clues = item["clues"]
        difficulty = item["difficulty"]

        puzzle_tokens = self.tokenizer(puzzle, padding="max_length", max_length=self.max_len, truncation=True)
        solution_tokens = self.tokenizer(solution, padding="max_length", max_length=self.max_len, truncation=True)

        clues_tensor = torch.tensor([int(i) for i in clues], dtype=torch.long).unsqueeze(0)
        difficulty_tensor = torch.tensor(difficulty, dtype=torch.float64).unsqueeze(0)

        return {
            "input_ids": puzzle_tokens["input_ids"].squeeze(),
            "attention_mask": puzzle_tokens["attention_mask"].squeeze(),
            "labels": solution_tokens["input_ids"].squeeze(),
            "clues": clues_tensor,
            "difficulty": difficulty_tensor
        }
