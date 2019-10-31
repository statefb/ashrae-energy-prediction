import numpy as np
import pandas as pd

from base import Feature

class IsWeekend(Feature):
    def create_feature(self):
        return (self.train["dayofweek"].apply(lambda x: True if x in (5, 6) else False),\
            self.test["dayofweek"].apply(lambda x: True if x in (5, 6) else False))

if __name__ == "__main__":
    IsWeekend().run()