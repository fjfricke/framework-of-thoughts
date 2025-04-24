import json
from pathlib import Path
from typing import Iterator

class HotpotQADatasetLoader(Iterator):
    def __init__(self, dataset_path: Path):
        data = json.loads(dataset_path.read_text())
        self.data = data
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
    dataset_path = Path.cwd().resolve() / "examples" / "hotpotqa" / "dataset" / "HotpotQA" / "hotpot_dev_fullwiki_v1.json"
    dataloader = HotpotQADatasetLoader(dataset_path)
    for i, batch in enumerate(dataloader):
        print(i, batch)
