import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
sys.path.append(".")

from .base import Model
from ..util import dump, load


class ModelLGBM(Model):
    """light GBM model class.
    """
    def __init__(self, run_fold_name: str, params: dict) -> None:
        super().__init__(run_fold_name, params)
        self.params["metric"] = set(params["metric"])

    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # データのセット
        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, label=tr_y)
        
        if validation:
            lgb_valid = lgb.Dataset(va_x, va_y)
            self.model = lgb.train(self.params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_valid),
                early_stopping_rounds=20,
                verbose_eval=20
            )
        else:
            self.model = lgb.train(self.params,
                lgb_train,
                num_boost_round=2000,
                early_stopping_rounds=20,
                verbose_eval=20
            )

    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def save_model(self):
        model_path = os.path.join('models', f'{self.run_fold_name}.model')
        # os.makedirs(os.path.dirname(model_path), exist_ok=True)
        dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join('models', f'{self.run_fold_name}.model')
        self.model = load(model_path)