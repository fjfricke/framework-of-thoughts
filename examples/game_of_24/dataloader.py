from enum import Enum
import pandas as pd
from typing import Iterator
from pathlib import Path

class Split(Enum):
    TRAIN = "training"
    TEST = "testing"

class GameOf24Dataloader(Iterator):
    def __init__(self, execution_mode: Split):
        self.df = pd.read_parquet(Path(__file__).parent / "dataset" / "train-00000-of-00001.parquet")
        self.df = self.df.sort_values("mean_time", ascending=True).reset_index(drop=True)
        if execution_mode == Split.TEST:
            self.df = self.df.iloc[901:1001].reset_index(drop=True)
        elif execution_mode == Split.TRAIN:
            self.df = pd.concat([self.df.iloc[801:901], self.df.iloc[1001:1101]], ignore_index=True)
        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")
        self.index = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {"input_list": self.df.iloc[idx]["numbers"].tolist()}, self.df.iloc[idx]["numbers"].tolist()

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.df):
            raise StopIteration
        item = self[self.index]
        self.index += 1
        return item

if __name__ == "__main__":
    dataloader = GameOf24Dataloader(execution_mode=Split.TEST)
    print(dataloader[0])