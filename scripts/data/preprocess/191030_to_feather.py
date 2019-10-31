import os
import pandas as pd
import sys
# sys.path.append("../../")

# read dataset as pandas.DataFrame
DATA_RAW_DIR = "data/raw/"
train = pd.read_csv(DATA_RAW_DIR + "train.csv")
test = pd.read_csv(DATA_RAW_DIR + "test.csv")
weather_train = pd.read_csv(DATA_RAW_DIR + "weather_train.csv")
weather_test = pd.read_csv(DATA_RAW_DIR + "weather_test.csv")
building_metadata = pd.read_csv(DATA_RAW_DIR + "building_metadata.csv")

# convert timestamp
train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"] = pd.to_datetime(test["timestamp"])
weather_train["timestamp"] = pd.to_datetime(weather_train["timestamp"])
weather_test["timestamp"] = pd.to_datetime(weather_test["timestamp"])

# save as feather format
DATA_FEATHER = "data/processed/"
train.to_feather(DATA_FEATHER + "train.feather")
test.to_feather(DATA_FEATHER + "test.feather")
weather_train.to_feather(DATA_FEATHER + "weather_train.feather")
weather_test.to_feather(DATA_FEATHER + "weather_test.feather")
building_metadata.to_feather(DATA_FEATHER + "building_meatadata.feather")