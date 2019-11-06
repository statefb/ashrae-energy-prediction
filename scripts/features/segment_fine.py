import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(".")

from scripts.features.base import Feature
from segment import DynpKcpd

class SegmentFine(Feature):
    def create_feature(self):
        n_train = self.train.shape[0]
        n_test = self.test.shape[0]

        # aggregate by day
        self.train.index = pd.to_datetime(self.train["timestamp"])
        self.test.index = pd.to_datetime(self.test["timestamp"])
        train_agg = self.train.groupby(\
            ["building_id", "meter", pd.Grouper(freq="24h")])\
            .agg(np.median).reset_index()
        test_agg = self.test.groupby(\
            ["building_id", "meter", pd.Grouper(freq="24h")])\
            .agg(np.median).reset_index()
        
        # KCPD for each building
        building_inds = train_agg["building_id"].unique()
        dfs_train = []; dfs_test = []
        for bidx in tqdm(building_inds):
            df_train = train_agg.query(f"building_id == {bidx}")
            df_test = test_agg.query(f"building_id == {bidx}")
            df_train = pd.DataFrame(dict(timestamp=\
                    pd.date_range(train_agg["timestamp"][0], train_agg["timestamp"][train_agg["timestamp"].size - 1])\
                ))\
                .merge(df_train, on="timestamp", how="left")
            df_test = pd.DataFrame(dict(timestamp=\
                    pd.date_range(test_agg["timestamp"][0], test_agg["timestamp"][test_agg["timestamp"].size - 1])\
                ))\
                .merge(df_test, on="timestamp", how="left")

            dfp = df_train.pivot(index="timestamp", columns="meter", values=["meter_reading"])
            X = dfp.values
            dfp_test = df_test.pivot(index="timestamp", columns="meter", values="building_id")
            dfp["timestamp"] = dfp.index
            dfp_test["timestamp"] = dfp_test.index
            # standardize
            sig = StandardScaler().fit_transform(X)
            # change-point detection
            kcpd = DynpKcpd(min_size=7, jump=1, max_n_bkps=50).fit(sig)
            bkps = kcpd.predict(sig, beta=0.1)["best_bkps"]
            bkps.insert(0, 0)
            # add segment label
            segment_label = np.repeat(range(len(bkps) - 1),\
                np.diff(np.array(bkps)))
            # import pdb; pdb.set_trace()

            dfp["segment"] = segment_label
            # NOTE: 2016 is leap year
            segment_label = np.delete(segment_label, 31 + 29 - 1)
            dfp_test["segment"] = segment_label.tolist() * 2

            df_train = df_train.merge(dfp["segment"], on="timestamp", how="left")
            df_test = df_test.merge(dfp_test["segment"], on="timestamp", how="left")
            dfs_train.append(df_train)
            dfs_test.append(df_test)

            del dfp
            del dfp_test
            del X
            del sig
            del kcpd
            del bkps
            del segment_label
            gc.collect()

        del df_train
        del df_test
        gc.collect()

        dfs_train = pd.concat(dfs_train, axis=0)
        dfs_train.index = dfs_train["timestamp"]
        dfs_train = dfs_train.groupby(["building_id", "meter"])\
            .resample("H").ffill()["segment"].reset_index()
        dfs_test = pd.concat(dfs_test, axis=0)
        dfs_test.index = dfs_test["timestamp"]
        dfs_test = dfs_test.groupby(["building_id", "meter"])\
            .resample("H").ffill()["segment"].reset_index()

        self.train.index.names = ["date"]
        self.test.index.names = ["date"]

        dfs_train["building_id"] = dfs_train["building_id"].astype(np.int16)
        dfs_train["meter"] = dfs_train["meter"].astype(np.int16)
        dfs_train["segment"] = dfs_train["segment"].fillna(method="ffill")
        dfs_test["building_id"] = dfs_test["building_id"].astype(np.int16)
        dfs_test["meter"] = dfs_test["meter"].astype(np.int16)
        dfs_test["segment"] = dfs_test["segment"].fillna(method="ffill")

        # merge segmentation label
        train = self.train.merge(dfs_train,\
            on=["building_id", "timestamp", "meter"],
            how="left")
        test = self.test.merge(dfs_test,\
            on=["building_id", "timestamp", "meter"],
            how="left")

        train["segment"] = train["segment"].fillna(method="ffill")
        test["segment"] = test["segment"].fillna(method="ffill")

        assert train.shape[0] == n_train, f"length must be the same. original:{n_train}, processed:{self.train.shape[0]}"
        assert test.shape[0] == n_test, f"length must be the same. original:{n_test}, processed:{self.test.shape[0]}"

        return train["segment"], test["segment"]


if __name__ == "__main__":
    SegmentFine().run()