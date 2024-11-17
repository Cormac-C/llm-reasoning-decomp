import pandas as pd
from torch.utils.data import Dataset


class Sudoku(Dataset):
    def __init__(self, data_file):
        self.dataset = pd.read_csv(data_file)

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

        # Return dictionary with question and solution
        return {"question": question, "answer": solution}
