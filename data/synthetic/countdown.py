from torch.utils.data import Dataset
import json

class Countdown(Dataset):
    def __init__(self, json_file, rating_threshold=0.97):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Filter examples with rating > threshold
        self.data = [item for item in self.data if item["rating"] > rating_threshold]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        nums = item["nums"]
        target = item["target"]
        solution = item["solution"]
        search_path  = item["search_path"]
        optimal_path = item["optimal_path"]


        question = (
            f"The target is: {target}\n"
            f"The available numbers are: {nums}\n"
            f"Describe how to reach the target using the given numbers."
        )

        answer = (
            f"The search path used for this problem was: {search_path}\n"
            f"The optimal path was: {optimal_path}\n"
            f"The final solution was: {solution}"
        )

        return {"question": question, "answer": answer}