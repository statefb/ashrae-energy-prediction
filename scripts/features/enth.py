import numpy as np
import pandas as pd
from CoolProp.HumidAirProp import HAPropsSI

from base import Feature

def calc_enth(df: pd.DataFrame) -> pd.Series:
    """Calculate enthalpy.
    """
    Tn = 273.15  # absolute temp.
    df_drop = df[["air_temperature", "dew_temperature", "sea_level_pressure"]].dropna()
    # NOTE: unit of temperature and pressure must be [K] and kPa respectively.
    enth = HAPropsSI("H",\
              "T", df_drop["air_temperature"].values + Tn,\
              "D", df_drop["dew_temperature"].values + Tn,\
              "P", df_drop["sea_level_pressure"].values * 10.0
             )
    enth = pd.Series(enth, index=df_drop.index)
    df = df.assign(enth=enth)
    return df["enth"]


class Enthalpy(Feature):
    def create_feature(self):
        weather_train = self.train.groupby(["site_id", "timestamp"]).mean().reset_index()
        weather_test = self.test.groupby(["site_id", "timestamp"]).mean().reset_index()

        weather_train["enth"] = calc_enth(weather_train)
        weather_test["enth"] = calc_enth(weather_test)

        self.train = self.train.merge(weather_train[["site_id", "timestamp", "enth"]],\
            left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"])
        self.test = self.test.merge(weather_test[["site_id", "timestamp", "enth"]],\
            left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"])
        
        return self.train["enth"], self.test["enth"]

class DescriptionMixin():
    def calc_description(self):
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

        # calculate representative values by each site_id and day
        weather_train.index = weather_train["timestamp"]
        weather_test.index = weather_test["timestamp"]
        enth_des_train = weather_train.groupby(["site_id", pd.Grouper(freq="24h")])\
            [["enth"]].agg([np.mean, max, min, np.median, np.std]).reset_index()
        enth_des_test = weather_test.groupby(["site_id", pd.Grouper(freq="24h")])\
            [["enth"]].agg([np.mean, max, min, np.median, np.std]).reset_index()
        
        # rename columns (convert multi index to single index)
        enth_des_train.columns = ['_'.join(col) if col[1] != "" else col[0] for col in enth_des_train.columns.values]
        enth_des_test.columns = ['_'.join(col) if col[1] != "" else col[0] for col in enth_des_test.columns.values]

        # upsample
        enth_des_train.index = pd.to_datetime(enth_des_train["timestamp"])
        enth_des_test.index = pd.to_datetime(enth_des_test["timestamp"])
        enth_des_train = enth_des_train.groupby("site_id").resample("H").ffill()\
            .drop(columns=["site_id", "timestamp"]).reset_index()
        enth_des_test = enth_des_test.groupby("site_id").resample("H").ffill()\
            .drop(columns=["site_id", "timestamp"]).reset_index()

        # merge
        self.train = self.train.merge(enth_des_train, on=["site_id", "timestamp"], how="left")
        self.test = self.test.merge(enth_des_test, on=["site_id", "timestamp"], how="left")
        
        assert self.train.shape[0] == n_train, f"length must be the same. original:{n_train}, processed:{self.train.shape[0]}"
        assert self.test.shape[0] == n_test, f"length must be the same. original:{n_test}, processed:{self.test.shape[0]}"

class EnthalpyMax(Feature, DescriptionMixin):
    def create_feature(self):
        self.calc_description()
        return self.train["enth_max"], self.test["enth_max"]

class EnthalpyMin(Feature, DescriptionMixin):
    def create_feature(self):
        self.calc_description()
        return self.train["enth_min"], self.test["enth_min"]

class EnthalpyMean(Feature, DescriptionMixin):
    def create_feature(self):
        self.calc_description()
        return self.train["enth_mean"], self.test["enth_mean"]

class EnthalpyMedian(Feature, DescriptionMixin):
    def create_feature(self):
        self.calc_description()
        return self.train["enth_median"], self.test["enth_median"]

class EnthalpyStd(Feature, DescriptionMixin):
    def create_feature(self):
        self.calc_description()
        return self.train["enth_std"], self.test["enth_std"]

if __name__ == "__main__":
    # Enthalpy().run()
    EnthalpyMax().run()
    EnthalpyMin().run()
    EnthalpyMean().run()
    EnthalpyMedian().run()
    EnthalpyStd().run()
