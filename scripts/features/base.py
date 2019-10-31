from abc import ABCMeta, abstractmethod
import pandas as pd
from os.path import join
from typing import Tuple

import sys
sys.path.append(".")
from scripts.util import timer, camel_to_snake

class Feature(metaclass=ABCMeta):
    """Feature base class.
    reference: https://amalog.hateblo.jp/entry/kaggle-feature-management
    """
    def __init__(self):
        self.name = camel_to_snake(self.__class__.__name__)

        # `train` and `test` must NOT be used as feature name
        assert self.name.find("train") == -1
        assert self.name.find("test") == -1

        self.load()  # load train/test dataset
        # path to save
        self.save_train_path = join("features/", f"{self.name}_train.pkl")
        self.save_test_path = join("features/", f"{self.name}_test.pkl")
    
    def run(self):
        """Create and save the feature as feather format.
        """
        with timer(self.name):
            f_train, f_test = self.create_feature()
        self.save(f_train, f_test)
        
    @abstractmethod
    def create_feature(self) -> Tuple[pd.Series, pd.Series]:
        raise NotImplementedError

    def load(self):
        self.train = pd.read_feather("data/processed/train.ftr")
        self.test = pd.read_feather("data/processed/test.ftr")
    
    def save(self, f_train: pd.Series, f_test: pd.Series):
        f_train.to_pickle(self.save_train_path)
        f_test.to_pickle(self.save_test_path)
