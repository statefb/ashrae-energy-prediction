from typing import Callable, List, Optional, Tuple, Union
from .base import Model
from .lgbm import ModelLGBM

def get_model_cls(model_name: str) -> Callable[[str, dict], Model]:
    d = {
        "lgbm": ModelLGBM
    }
    return d[model_name]