import numpy as np
import pandas as pd
import gc

from base import  Feature


class LagDescriptionMixin():
    def calc_lag_description(self):
        # for assertion
        n_train = self.train.shape[0]
        n_test = self.test.shape[0]

        # load enthalpy (pre-computed)
        enth_train = pd.read_pickle("features/enthalpy_train.pkl")
        enth_test = pd.read_pickle("features/enthalpy_test.pkl")

        self.train["enth"] = enth_train
        self.test["enth"] = enth_test

        weather_train = self.train.groupby(["site_id", "timestamp"]).mean().reset_index()
        weather_test = self.test.groupby(["site_id", "timestamp"]).mean().reset_index()

        self.train = self.train[["site_id", "timestamp"]]
        self.test = self.test[["site_id", "timestamp"]]

        # calculate representative values by each site_id and day
        weather_train.index = weather_train["timestamp"]
        weather_test.index = weather_test["timestamp"]
        enth_des_train = weather_train.groupby(["site_id", pd.Grouper(freq="24h")])\
            [["enth"]].agg([np.mean, max, min, np.median, np.std]).reset_index()
        enth_des_test = weather_test.groupby(["site_id", pd.Grouper(freq="24h")])\
            [["enth"]].agg([np.mean, max, min, np.median, np.std]).reset_index()

        del weather_train
        del weather_test
        gc.collect()
        
        enth_des_train.index = pd.to_datetime(enth_des_train["timestamp"])
        enth_des_test.index = pd.to_datetime(enth_des_test["timestamp"])
        enth_des_train.index.names = ["date"]
        enth_des_test.index.names = ["date"]

        # rename columns (convert multi index to single index)
        enth_des_train.columns = ['_'.join(col) if col[1] != "" else col[0] for col in enth_des_train.columns.values]
        enth_des_test.columns = ['_'.join(col) if col[1] != "" else col[0] for col in enth_des_test.columns.values]
        # create lag
        enth_des_train = enth_des_train.merge(\
            enth_des_train.drop(columns=["timestamp"]).shift(1, freq="D").rename(columns=lambda x: x + "_p1d" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        ).merge(\
            enth_des_train.drop(columns=["timestamp"]).shift(7, freq="D").rename(columns=lambda x: x + "_p1w" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        ).merge(\
            enth_des_train.drop(columns=["timestamp"]).shift(-1, freq="D").rename(columns=lambda x: x + "_m1d" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        ).merge(\
            enth_des_train.drop(columns=["timestamp"]).shift(-7, freq="D").rename(columns=lambda x: x + "_m1w" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        )

        enth_des_test = enth_des_test.merge(\
            enth_des_test.drop(columns=["timestamp"]).shift(1, freq="D").rename(columns=lambda x: x + "_p1d" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        ).merge(\
            enth_des_test.drop(columns=["timestamp"]).shift(7, freq="D").rename(columns=lambda x: x + "_p1w" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        ).merge(\
            enth_des_test.drop(columns=["timestamp"]).shift(-1, freq="D").rename(columns=lambda x: x + "_m1d" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        ).merge(\
            enth_des_test.drop(columns=["timestamp"]).shift(-7, freq="D").rename(columns=lambda x: x + "_m1w" if x not in ("site_id", "timestamp") else x),\
            on=["site_id", "date"], how="left"\
        )

        # upsample
        enth_des_train = enth_des_train.groupby("site_id").resample("H").ffill()\
            .drop(columns=["site_id", "timestamp"]).reset_index()
        enth_des_test = enth_des_test.groupby("site_id").resample("H").ffill()\
            .drop(columns=["site_id", "timestamp"]).reset_index()

        # merge
        self.train = self.train.merge(enth_des_train, left_on=["site_id", "timestamp"],\
            right_on=["site_id", "date"], how="left")
        self.test = self.test.merge(enth_des_test, left_on=["site_id", "timestamp"],\
            right_on=["site_id", "date"], how="left")
        
        assert self.train.shape[0] == n_train, f"length must be the same. original:{n_train}, processed:{self.train.shape[0]}"
        assert self.test.shape[0] == n_test, f"length must be the same. original:{n_test}, processed:{self.test.shape[0]}"

class EnthalpyMaxPlus1day(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_max_p1d"], self.test["enth_max_p1d"]

class EnthalpyMaxMinus1day(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_max_m1d"], self.test["enth_max_m1d"]

class EnthalpyMaxPlus1week(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_max_p1w"], self.test["enth_max_p1w"]

class EnthalpyMaxMinus1week(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_max_m1w"], self.test["enth_max_m1w"]

class EnthalpyMinPlus1day(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_min_p1d"], self.test["enth_min_p1d"]

class EnthalpyMinMinus1day(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_min_m1d"], self.test["enth_min_m1d"]

class EnthalpyMinPlus1week(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_min_p1w"], self.test["enth_min_p1w"]

class EnthalpyMinMinus1week(Feature, LagDescriptionMixin):
    def create_feature(self):
        self.calc_lag_description()
        return self.train["enth_min_m1w"], self.test["enth_min_m1w"]

if __name__ == "__main__":
    EnthalpyMaxPlus1day().run()
    EnthalpyMaxMinus1day().run()
    EnthalpyMaxPlus1week().run()
    EnthalpyMaxMinus1week().run()
    EnthalpyMinPlus1day().run()
    EnthalpyMinMinus1day().run()
    EnthalpyMinPlus1week().run()
    EnthalpyMinMinus1week().run()