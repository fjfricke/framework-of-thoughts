from pathlib import Path
from collections.abc import Iterator
from typing import Iterable

class TestDatasetLoader(Iterator):
    def __init__(self, file_path):
        # Read the file and parse the data
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                # Split the line into input and output
                input_str, output_str = line.strip().split(', ')
                # Evaluate the input expression and convert output to int
                output_value = int(output_str)
                self.data.append((input_str, output_value))
        self.index = 0  # Initialize the index for iteration

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return self  # An iterator must return itself

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration  # End of iteration
        input_str, output_value = self.data[self.index]
        self.index += 1
        return {"start": input_str}, output_value
    
class TestDatasetLoaderWithYield(Iterable):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                # Split the line into input and output
                input_str, output_str = line.strip().split(', ')
                # Yield the parsed input and output
                yield {"start": input_str}, int(output_str)

# Example usage
if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "test_dataset.txt"
    dataloader = TestDatasetLoaderWithYield(dataset_path)
    for batch in dataloader:
        print(batch) 
    dataloader = TestDatasetLoader(dataset_path)
    for batch in dataloader:
        print(batch) 