from functools import partial

from latex2sympy2 import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm

from verl.utils.reward_score.ttrl.mme import extract_mme_answer
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer


def auto_extract(task, all_outputs, extra_info=None):
    task2extract_fn = {
        "math": partial(extract_answer, data_name=task),
        "gpqa": partial(extract_answer, data_name=task),
        "bbox": partial(extract_answer, data_name=task),
        "mme": None,
    }
    assert task in task2extract_fn, f"{task} not in {list(task2extract_fn.keys())}"
    extract_fn = task2extract_fn[task]

    if task == "mme":
        extra_info = extra_info or [{} for _ in all_outputs]
        model_answers = [
            extract_mme_answer(generated_text, row_extra_info)
            for generated_text, row_extra_info in zip(all_outputs, extra_info)
        ]
    else:
        model_answers = [extract_fn(generated_text) for generated_text in all_outputs]

    return [answer if answer is not None else "" for answer in model_answers]
