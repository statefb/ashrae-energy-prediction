import numpy as np
import pandas as pd

from base import Feature

class IsHoliday(Feature):
    def create_feature(self):
        # reference: https://www.kaggle.com/rohanrao/ashrae-half-and-half
        holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]

        self.train["timestamp"] = pd.to_datetime(self.train["timestamp"], format="%Y-%m-%d %H:%M:%S")
        self.test["timestamp"] = pd.to_datetime(self.test["timestamp"], format="%Y-%m-%d %H:%M:%S")

        tr_is_holiday = (self.train["timestamp"].isin(holidays)).astype(bool)
        te_is_holiday = (self.test["timestamp"].isin(holidays)).astype(bool)

        return tr_is_holiday, te_is_holiday

if __name__ == "__main__":
    IsHoliday().run()