import numpy as np
import pandas as pd

from base import Feature

class IsWeekend(Feature):
    def create_feature(self):
        tr_is_weekend = self.train["dayofweek"].apply(lambda x: True if x in (5, 6) else False).rename("is_weekend")
        te_is_weekend = self.test["dayofweek"].apply(lambda x: True if x in (5, 6) else False).rename("is_weekend")

        return tr_is_weekend, te_is_weekend

if __name__ == "__main__":
    IsWeekend().run()