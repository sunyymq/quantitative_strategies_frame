import copy
import sys
import pandas as pd
import numpy as np
import os
from itertools import chain
from functools import reduce
import statsmodels.api as sm
import shelve


# 均线
def ma(dat_df, n, axis=0):

    if n == 0:
        return dat_df

    if axis == 1:
        dat_df = dat_df.T

    dat_v = dat_df.values
    ma_v = np.full(dat_df.shape, np.nan)
    if len(dat_df.columns) > 1:
        for i in range(n, len(dat_df.columns)):
            count_sec = dat_v[:, i - n:i].mean(axis=1)
            ma_v[:, i] = count_sec
    else:
        for i in range(n, len(dat_df.index)):
            count_sec = dat_v[i - n:i, :].mean(axis=0)
            ma_v[i, :] = count_sec

    ma_df = pd.DataFrame(data=ma_v, index=dat_df.index, columns=dat_df.columns)

    return ma_df
