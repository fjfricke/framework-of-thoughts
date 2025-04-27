import ast
import pandas as pd
from pathlib import Path
from typing import Iterator

class SortingDataloader(Iterator):
    def __init__(self, dataset_path: Path):
        data = pd.read_csv(dataset_path)
        self.data = data
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
