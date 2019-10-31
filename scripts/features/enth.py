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

if __name__ == "__main__":
    Enthalpy().run()
