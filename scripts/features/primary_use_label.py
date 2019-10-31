import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from base import Feature

class PrimaryUseLabel(Feature):
    def create_feature(self):
        le = LabelEncoder()
        self.train["primary_use"] = le.fit_transform(self.train["primary_use"])
        self.test["primary_use"] = le.transform(self.test["primary_use"])
        
        return self.train["primary_use"], self.test["primary_use"]

if __name__ == "__main__":
    PrimaryUseLabel().run()