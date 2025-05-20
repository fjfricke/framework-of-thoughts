from enum import Enum
import json
from pathlib import Path
import random
from typing import Iterator

class Split(Enum):
    TRAIN = "training"
    VALIDATION = "validation"

class HotpotQADatasetLoader(Iterator):
    def __init__(self, execution_mode: Split, dataset_path: Path, split: float = 0.8, seed: int = 42, total_size: int = None):
        try:
            data = json.loads(dataset_path.read_text())
        except Exception:
            data = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        random_generator = random.Random(seed)
        random_generator.shuffle(data)
        if not (0.0 < split <= 1.0):
            raise ValueError("Split must be a float between 0 and 1.")
        if execution_mode not in [Split.TRAIN, Split.VALIDATION]:
            raise ValueError("Invalid execution mode. Choose 'training' or 'validation'.")
        if total_size is not None:
            data = data[:total_size]
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
        return {"question": self.data[idx]["question"]}, self.data[idx]["answer"]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        item = self[self.index]
        self.index += 1
        return item


if __name__ == "__main__":
    # dataset_path = Path.cwd().resolve() / "examples" / "hotpotqa" / "dataset" / "HotpotQA" / "hotpot_dev_fullwiki_v1.json"
    dataset_path = Path.cwd().resolve() / "examples" / "hotpotqa" / "dataset" / "HotpotQA" / "musique_full_v1.0_dev.jsonl"
    dataloader = HotpotQADatasetLoader(dataset_path=dataset_path, execution_mode=Split.VALIDATION, split=0.5, seed=42, total_size=2000)
    for i, batch in enumerate(dataloader):
        print(i, batch)
