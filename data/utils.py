import os
import sys

from datasets import Dataset
from torch.utils.data import Subset

# Setup module path for local imports
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from data.sudoku import Sudoku
from data.format import (
    chat_format_qa_instance,
    lm_format_qa_instance,
    chat_create_fewshot_prompt,
)


def load_prep_sudoku_dataset(
    tokenizer, instruction_tuned=True, few_shot=None, test_split_size=0.2
):
    dataset = Sudoku(data_file=os.environ["SUDOKU_PATH"])

    if instruction_tuned:
        formatted_list = []
        if few_shot is not None:
            # Pick examples for few-shot learning
            fewshot_examples = Subset(dataset, range(few_shot))
            dataset = Subset(dataset, range(few_shot, len(dataset)))
            print(f"Few-shot examples: {len(fewshot_examples)}")
            print(f"Remaining examples: {len(dataset)}")

            formatted_list = [
                chat_create_fewshot_prompt(
                    example, examples=fewshot_examples, num_shots=few_shot
                )
                for example in dataset
            ]
        else:
            formatted_list = [chat_format_qa_instance(example) for example in dataset]
        formatted_list = tokenizer.apply_chat_template(
            formatted_list, tokenize=False, add_generation_prompt=False
        )

    else:
        formatted_list = [lm_format_qa_instance(example) for example in dataset]

    num_clues_list = [example["num_clues"] for example in dataset]

    dataset = Dataset.from_dict(
        {"formatted_text": formatted_list, "num_clues": num_clues_list}
    )

    dataset = dataset.train_test_split(test_size=test_split_size)

    return dataset
