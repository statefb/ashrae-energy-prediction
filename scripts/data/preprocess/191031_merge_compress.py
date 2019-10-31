"""Merge & compress dataframe.
"""
import numpy as np
import pandas as pd
import sys
sys.path.append(".")
from scripts.util import timer, reduce_mem_usage

with timer("reading feather data"):
    DATA_FEATHER = "data/processed/"
    train = pd.read_feather(DATA_FEATHER + "train.feather")
    test = pd.read_feather(DATA_FEATHER + "test.feather")
    weather_train = pd.read_feather(DATA_FEATHER + "weather_train.feather")
    weather_test = pd.read_feather(DATA_FEATHER + "weather_test.feather")
    building_metadata = pd.read_feather(DATA_FEATHER + "building_metadata.feather")

print(train.head())

with timer("merging"):
    train = train.merge(building_metadata, left_on="building_id", right_on="building_id", how="left")
    train = train.merge(weather_train, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"])
    test = test.merge(building_metadata, left_on="building_id", right_on="building_id", how="left")
    test = test.merge(weather_test, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"])

print(train.head())

del building_metadata
del weather_train
del weather_test

with timer("unravel timestmap"):
    train["hour"] = train["timestamp"].dt.hour
    train["day"] = train["timestamp"].dt.day
    train["month"] = train["timestamp"].dt.month
    train["dayofweek"] = train["timestamp"].dt.dayofweek
    test["hour"] = test["timestamp"].dt.hour
    test["day"] = test["timestamp"].dt.day
    test["month"] = test["timestamp"].dt.month
    test["dayofweek"] = test["timestamp"].dt.dayofweek

print(train.head())

with timer("adding meter_label"):
    meter = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
    train = train.assign(
        meter_label=lambda df: df["meter"].map(meter)
    )
    test = test.assign(
        meter_label=lambda df: df["meter"].map(meter)
    )

print(train.head())

# add log1p y
train["log_meter_reading"] = np.log1p(train["meter_reading"])


with timer("reduce memory"):
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

with timer("saving"):
    train.to_feather(DATA_FEATHER + "train.ftr")
    test.to_feather(DATA_FEATHER + "test.ftr")
