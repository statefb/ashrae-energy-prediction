import json
import numpy as np
import pandas as pd
import sys
sys.path.append(".")

from scripts.models import get_model_cls
from scripts.runner import Runner
from scripts.util import create_submission

if __name__ == '__main__':
    CONFIG_FILE = "configs/baseline.json"
    RUN_NAME = "simple_lgbm"

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    runner = Runner(RUN_NAME, get_model_cls(config["model_name"]),\
        config["features"], config["target"], config["hyper_params"], config["loss"])
    runner.run_train_cv()
    runner.run_predict_cv()
    create_submission(RUN_NAME)

    # # xgboostによる学習・予測
    # runner = Runner('xgb1', ModelXGB, features, params_xgb)
    # runner.run_train_cv()
    # runner.run_predict_cv()
    # Submission.create_submission('xgb1')

    # # ニューラルネットによる学習・予測
    # runner = Runner('nn1', ModelNN, features, params_nn)
    # runner.run_train_cv()
    # runner.run_predict_cv()
    # Submission.create_submission('nn1')

    '''
    # (参考）xgboostによる学習・予測 - 学習データ全体を使う場合
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_predict_all()
    Submission.create_submission('xgb1-train-all')
    '''