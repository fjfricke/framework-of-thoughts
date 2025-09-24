# Framework-of-Thoughts

## Description
`framework-of-thoughts` is a Python library designed to model and optimize graph-based multi-step reasoning problems using large language models (LLMs). It provides tools for creating, visualizing and executing prompting strategies, and optimizing hyperparameters using optuna.

## Installation

To install the project, you can use `pip install git+https://github.com/fjfricke/llm-graph-optimizer.git`. Ensure you have Python 3.12 installed.

## Development

To install the project for development purposes or to run the examples, ensure you have Python 3.12 installed. Clone the project. Then, use `poetry` to set up the environment:

```bash
poetry install
```

This will install all required dependencies.

### Activating the Virtual Environment

After installing the dependencies, activate the virtual environment created by `poetry`:

```bash
poetry shell
```

## Optional Features

The project provides several optional features that can be installed using `poetry` extras. These extras enable additional functionality for specific use cases:

- **examples**: Includes dependencies needed to run the examples in the `examples` directory. If you want to replicate the studies, please also install the `optimizer` extra.
- **dataset**: Adds support to run experiments on entire datasets.
- **optimizer**: `dataset` + installs `optuna` and `optuna-dashboard` for hyperparameter optimization, as well as `dspy`.
- **dev**: Installs development tools like `ruff` and `pre-commit` for linting and code formatting if you want to contribute to the project. :)

### Installing Extras

To install an extra, use the following command:

```bash
poetry install --extras "<extra_name>"
```

For example, to install the `examples` extra:

```bash
poetry install --extras "examples"
```

### Installing All Extras

To install all extras at once, use the following command:

```bash
poetry install --all-extras
```

## Usage

Explore the `examples` directory for modelling Tree-of-Thought on the Sorting Problem and Game of 24, ProbTree and Graph-of-Thought on the NDA task.

A mock example called "test" is also provided to show how to use the library using Self-Consistency prompting on simple math problems.

A tutorial is provided under `examples/test/tutorial.ipynb` to show how to use the library for modelling and optimizing execution graphs. I recommend checking that out to get started. You can find a snapshot html version at `examples/test/tutorial_snapshot.html`.

## Features

- Modelling of graph-based prompting strategies with support for various graph operations and measurements
- Integration with `optuna` for optimization of hyperparameters
- Example scripts and datasets to reproduce the results in the paper.

## Setting the OpenAI API key

Please add an `.env` file to the root of the project with the following:

```
OPENAI_API_KEY=<your_openai_api_key>
```

Alternatively, you can also pass the api key directly to the LLM constructor.

```python
llm = OpenAIChat(api_key="your_openai_api_key", ...) or
llm_with_logprobs = OpenAIChatWithLogprobs(api_key="your_openai_api_key", ...)
```

# Disclaimer

For the development, Cursor, mainly with Openai GPT 4o, was used; mainly for code completion, repository structuring, refactoring, and error fixing.