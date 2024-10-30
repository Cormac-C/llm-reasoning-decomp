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

        # Format the puzzle with the answer
        question = f"Given the Sudoku puzzle {puzzle}, which has {clues} clues and a difficulty rating of {difficulty}. Please solve for the final arrangement."
        input_text = f"{question} ### Answer: {solution}"

        tokens = self.tokenizer(input_text, padding="longest", return_tensors="pt")

        # Return input_ids, attention_mask, clues, and difficulty
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "clues": torch.tensor(int(clues), dtype=torch.long),
            "difficulty": torch.tensor(float(difficulty), dtype=torch.float32)
        }
