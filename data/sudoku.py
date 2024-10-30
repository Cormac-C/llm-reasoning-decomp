import pandas as pd
import torch
from torch.utils.data import Dataset

class Sudoku(Dataset):
    def __init__(self, data_file, tokenizer):
        self.dataset = pd.read_csv(data_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        puzzle = item["puzzle"]
        solution = item["solution"]
        clues = item["clues"]
        difficulty = item["difficulty"]

        question = f"Given a Sudoku puzzle, {puzzle}. Please solve for the final arrangement."

        input_text = f"{question} {solution}"
        tokens = self.tokenizer(input_text, padding="longest", return_tensors="pt")

        clues_tensor = torch.tensor([int(i) for i in clues], dtype=torch.long).unsqueeze(0)
        difficulty_tensor = torch.tensor(difficulty, dtype=torch.float64).unsqueeze(0)

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": puzzle_tokens["attention_mask"].squeeze(),
            "clues": clues_tensor,
            "difficulty": difficulty_tensor
        }
