# Training Reasoning Sub-Skills in LLMs with Synthetic Data

## Requirements

To install requirements:

```setup env
conda env create -f environment.yml

conda activate reasoning-decomp
```

To run scripts you will also need to set your Huggingface token and WANDB key for monitoring and recording results:

```setup tokens
export HF_TOKEN= <your token>

export WANDB_KEY= <your token>
```

To run training scripts and save the adapters, you will need to set the base directory where each adapter wil be saved as an environment variable

```setup base_dir
export BASE_DIR= <your local base dir>
```

If you want to run the sudoku or countdown scripts then you will need to load the sudoku and countdown datasets locally then set the local paths as environment variables

```setup data
export SUDOKU_PATH= <your local path>

export COUNTDOWN_PATH= <your local path>
```

## Training

To train the LORA adapters we have scripts for each dataset:

```train scripts
python scripts/train_baseline_zebra.py

python scripts/train_baseline_sudoku.py

python scripts/train_countdown_sos.py
```

For each script you can specify the size of the base model with the flag `--base_model= <1B or 3B>`. For example:

```train scripts args
python scripts/train_baseline_zebra.py --base_model=1B
```

will fine-tune an adapter on top of Llama-3.2-1B-Instruct.

## Evaluation

To evaluate the base model and the trained adapters we have the following scripts:

```eval scripts
python scripts/eval_base_zebra.py

python scripts/eval_base_sudoku.py

python scripts/eval_adapter_zebra.py <adapter_dir>

python scripts/eval_adapter_sudoku.py <adapter_dir>

python scripts/eval_sos_zebra.py <adapter_dir>

python scripts/eval_sos_sudoku.py <adapter_dir>
```

For the adapter and sos adapter scripts you are required to specify the path to the adapter which was trained using the above scripts as a positional argument.

For each script you can specify the size of the base model with the flag `--base_model= <1B or 3B>` and specify the number of few-shot examples with the flag `--few_shot= <int>`. For example:
For each eval script you can specify the number of few-shot examples as a parameter, the default is zero-shot. For example:

```eval scripts few shot
python scripts/eval_base_zebra.py --few_shot=3 --base_model=1B
```

will evaluate Llama-3.2-1B-Instruct on the zebra test set with 3 few shot examples.

## Data

- The Zebra dataset is downloaded directly from [Huggingface](https://huggingface.co/datasets/allenai/ZebraLogicBench).
- The Sudoku dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings/data).

To generate the Countdown Dataset:

#### Prerequisites

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/kanishkg/stream-of-search.git
   cd stream-of-search

   ```

2. **Set Up the Environment**:

- Install Miniconda (if not already installed):
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  ```
- Create and activate a Conda environment:
  ```bash
  conda create -n sos python=3.11
  conda activate sos
  ```
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

3. **Generating the Countdown Dataset**:
   - Run the Script to Generate the Dataset:
     ```bash
      sh scripts/gen_task.sh
     ```
   - This will generate the Countdown dataset
   - Save this as countdown.json in the appropriate output directory
   - This file will contain structured examples for the Countdown problem. Each entry includes the target number, available numbers, the optimal solution path, the whole search trajectory and a rating field.
