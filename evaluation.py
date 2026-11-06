import ast
import random
import re

import numpy as np


def _flatten_answer(value):
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_answer(item))
        return flattened
    return [value]


def _canonical_answer_atom(value):
    text = " ".join(str(value).strip().split())
    lower = text.lower()
    if lower in ("false", "no", "none"):
        return "0"
    if lower in ("true", "yes"):
        return "1"
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text.casefold()


def normalize_answer_units(ans):
    if isinstance(ans, str):
        raw = ans.strip()
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = [item for item in raw.split(", ") if item] if ", " in raw else raw
    else:
        parsed = ans
    return [
        _canonical_answer_atom(item)
        for item in _flatten_answer(parsed)
        if str(item).strip()
    ]


def judge(pred, ans):
    expected = normalize_answer_units(ans)
    observed = normalize_answer_units(pred)
    return bool(expected) and sorted(observed) == sorted(expected)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def extract_prediction(logs_text):
    marker = "Final Executed Answer ---> "
    if marker not in logs_text:
        return ""
    prediction = logs_text.rsplit(marker, 1)[1]
    return prediction.split("\n----------------------------------------------------------\n", 1)[0].strip()
