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

To evaluate the base model and the trained adapeers we have the following scripts:

```eval scripts
python scripts/eval_base_zebra.py

python scripts/eval_base_sudoku.py

python scripts/eval_adapter_zebra.py

python scripts/eval_adapter_sudoku.py

python scripts/eval_sos_zebra.py

python scripts/eval_sos_sudoku.py
```

For each script you can specify the size of the base model with the flag `--base_model= <1B or 3B>` and specify the number of few-shot examples with the flag `--few_shot= <int>`. For example:
For each eval script you can specify the number of few-shot examples as a parameter, the default is zero-shot. For example:

```eval scripts few shot
python scripts/eval_base_zebra.py --few_shot=3 --base_model=1B
```

will evaluate Llama-3.2-1B-Instruct on the zebra test set with 3 few shot examples.

## Data

TODO: add details about generating countdown dataset

## Pre-trained adapters

TODO: maybe download and host some of the pre-trained adapters somewhere?
