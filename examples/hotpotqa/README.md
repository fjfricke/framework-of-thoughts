# HotpotQA

## Setup

In order to load the dataset, you need to run:

```bash
python services/download_dataset.py
```

Then, in order to index the dataset, you need to run:

```bash
python services/index_hotpotqa.py
```
## Operations

The `programs` folder contains the code for the different prompting strategies that are used to solve the HotpotQA dataset.

`programs/operations` contains the operations defined specifically to model ProbTree: specifically the `reasoning` and `understanding` subfolders.

## Execution

To run ProbTree on one example, please refer to the `probtree.py` file.
To run a dataset evaluation or an optimization study, please refer to the `probtree_study.py` file and change the function in the `if __name__ == "__main__":` block.

## Output
The results of the study as well as the cache are saved in the `output` folder.