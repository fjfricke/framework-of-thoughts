# HotpotQA

## Setup

## Wiki dump and HotpotQA/MuSiQue Dataset

In order to load the wikipedia dump ad the HotpotQA dataset, you need to run:

```bash
python services/download_dataset.py
```

Then, in order to index the dataset, you need to run:

```bash
python services/index_hotpotqa.py
```

MuSiQue can be downloaded from the [musique repository](https://github.com/StonyBrookNLP/musique). See instructions there. In our case, the `musique_full_v1.0_dev.jsonl` file was only used. Place it also in the `dataset/HotpotQA` folder. We used the October 2017 Wikipedia dump for both retrievers.

## Operations

The `programs` folder contains the code for the different prompting strategies that are used to solve the HotpotQA dataset.

`programs/operations` contains the operations defined specifically to model ProbTree: specifically the `reasoning` and `understanding` subfolders.

## Execution

To run ProbTree on one example, please refer to the `probtree.py` file.
To run a dataset evaluation or an optimization study, please refer to the `dataset_evaluation.py` file and change the parameter in the `if __name__ == "__main__":` block to the correct dataset.

## Output
The results of the study are saved in the `output` folder.