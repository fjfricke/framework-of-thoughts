from enum import Enum
from pathlib import Path
from typing import Iterator
import pandas as pd

class Split(Enum):
    TRAIN = "training"
    TEST = "testing"


class DocMergeDataloader(Iterator):
    def __init__(self, execution_mode: Split, dataset_path: Path, split: float = 0.5, seed: int = 42):
        self.dataset_path = dataset_path
        data = pd.read_csv(dataset_path)
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        if not (0.0 < split <= 1.0):
            raise ValueError("Split must be a float between 0 and 1.")
        if execution_mode not in [Split.TRAIN, Split.TEST]:
            raise ValueError("Invalid execution mode. Choose 'training' or 'testing'.")
        
        split_index = int(len(data) * split)
        if execution_mode == Split.TRAIN:
            self.data = data[:split_index]
        else:  # test
            self.data = data[split_index:]
        
        self.execution_mode = execution_mode
        self.index = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"docs": [self.data.iloc[idx]["document1"], self.data.iloc[idx]["document2"], self.data.iloc[idx]["document3"], self.data.iloc[idx]["document4"]]}, None

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        item = self[self.index]
        self.index += 1
        return item


if __name__ == "__main__":
    dataloader = DocMergeDataloader(Path.cwd().resolve() / "examples" / "doc_merge" / "dataset" / "documents.csv")
    for i, batch in enumerate(dataloader):
        print(i, batch)
    print(len(dataloader))