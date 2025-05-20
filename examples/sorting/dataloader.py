import ast
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Iterator

class Split(Enum):
    TRAIN = "training"
    VALIDATION = "validation"


class SortingDataloader(Iterator):
    def __init__(self, execution_mode: Split, dataset_path: Path, split: float = 0.8, seed: int = 42):
        data = pd.read_csv(dataset_path)
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        if not (0.0 < split <= 1.0):
            raise ValueError("Split must be a float between 0 and 1.")
        if execution_mode not in [Split.TRAIN, Split.VALIDATION]:
            raise ValueError("Invalid execution mode. Choose 'training' or 'validation'.")
        
        split_index = int(len(data) * split)
        if execution_mode == Split.TRAIN:
            self.data = data[:split_index]
        else:  # validation
            self.data = data[split_index:]
        
        self.execution_mode = execution_mode
        self.index = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_list": ast.literal_eval(self.data.iloc[idx]["Unsorted"]), "expected_output": ast.literal_eval(self.data.iloc[idx]["Sorted"])}, ast.literal_eval(self.data.iloc[idx]["Sorted"])
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        item = self[self.index]
        self.index += 1
        return item


if __name__ == "__main__":
    dataset_path = Path.cwd().resolve() / "examples" / "sorting" / "dataset" / "sorting_064.csv"
    dataloader = SortingDataloader(dataset_path)
    for i, batch in enumerate(dataloader):
        print(i, batch)
