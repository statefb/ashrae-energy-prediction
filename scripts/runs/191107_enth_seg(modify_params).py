import json
import numpy as np
import pandas as pd
import sys
sys.path.append(".")

from scripts.models import get_model_cls
from scripts.runner import Runner

if __name__ == '__main__':
    CONFIG_FILE = "configs/enth_and_seg(modify_params).json"
    RUN_NAME = "enth_and_seg(modify_params)"

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    runner = Runner(RUN_NAME, get_model_cls(config["model_name"]),\
        config["features"], config["target"], config["hyper_params"], config["loss"])

    runner.run_train_cv()
    runner.run_predict_cv()
    runner.create_submission()