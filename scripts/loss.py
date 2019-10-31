import numpy as np
from sklearn.metrics import mean_squared_error

def get_loss_func(loss_name):
    d = {
        "rmse": rmse
    }
    return d[loss_name]

def rmse(act, pred):
    return np.sqrt(mean_squared_error(act, pred))