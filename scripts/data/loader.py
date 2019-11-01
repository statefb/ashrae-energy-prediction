from os.path import basename, splitext
from typing import List, Tuple
import pandas as pd
from glob import glob

import sys
sys.path.append(".")
from scripts.util import timer

def get_filename(path: str) -> str:
    filename = splitext(basename(path))[0]
    # exclude `_train` and `_test`
    filename = "_".join(filename.split("_")[:-1])
    return filename

def load_datasets(feats: List[str], y_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # load feather files
    train = pd.read_feather("data/processed/train.ftr")
    test = pd.read_feather("data/processed/test.ftr")
    
    # filtering by given feature list
    filtered_columns = train.columns[train.columns.isin(feats)].tolist()
    test = test[filtered_columns]
    filtered_columns.append(y_name)  # add objective variable
    train = train[filtered_columns]
    import pdb; pdb.set_trace()
    # load FE files
    fe_files = glob("features/*.pkl")
    for fe_file in fe_files:
        if get_filename(fe_file) not in feats:
            # skip if not exist
            continue
        # load data
        x = pd.read_pickle(fe_file)

        # add FE feature to train/test
        if fe_file.find("train") != -1:
            train[x.name] = x
        else:
            test[x.name] = x

    return train, test


class DataLoader():
    def __init__(self, feats, y_name):
        with timer("loading data"):
            self.train, self.test = load_datasets(feats, y_name)
        self.feats = feats
        self.y_name = y_name

    @property
    def x_train(self):
        return self.train.drop(columns=self.y_name)

    @property
    def y_train(self):
        return self.train[self.y_name]

    @property
    def x_test(self):
        return self.test
    