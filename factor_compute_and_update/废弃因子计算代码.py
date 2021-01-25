import copy
import sys
import pandas as pd
import numpy as np
import os
from itertools import chain
from functools import reduce
import statsmodels.api as sm
import shelve
from datetime import datetime, timedelta
# from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from utility.tool0 import Data, scaler
from utility.factor_data_preprocess import add_to_panels, align
from utility.relate_to_tushare import generate_months_ends, trade_days
from utility.tool1 import CALFUNC, _calculate_su_simple, parallelcal,  lazyproperty, time_decorator, \
    get_signal_season_value, get_fill_vals, linear_interpolate, get_season_mean_value
from utility.constant import data_dair, root_dair
from utility.tool3 import adjust_months, add_to_panels, append_df


START_YEAR = 2009

class Factor_Compute(CALFUNC):

    def __init__(self, status):
        super().__init__()
        # status = 'update' 表示仅对已有的因子值进行更新， ='all' 表示全部重新计算
        self._mes = generate_months_ends()
        self._status = status

    def _get_update_month(self, fn):
        factor_m = eval('self.' + fn)
        # factor_m = self.RETURN_12M
        last_dt = pd.to_datetime(factor_m.columns[-1])
        to_update_month_list = [i for i in self._mes if i > last_dt]
        if len(to_update_month_list) == 0:
            print('没有更新必要')
            return None
            # sys.exit()
        else:
            return to_update_month_list

    @lazyproperty
    def rsi(self):
        pct = self.changepct_daily.T
        pct.index = pd.to_datetime(pct.index)
        pct = pct / 100
        pct_nonnegative = copy.deepcopy(pct)
        pct_nonnegative[pct_nonnegative < 0] = 0
        numerator = pct_nonnegative.rolling(20).sum()
        pct_positive = abs(pct)
        denominator = pct_positive.rolling(20).sum()
        RSI = (numerator / denominator) * 100
        RSI = RSI.T
        RSI = CALFUNC.del_dat_early_than(RSI, START_YEAR)
        return RSI


    @lazyproperty
    def psy(self):
        pct = self.changepct_daily.T
        pct.index = pd.to_datetime(pct.index)
        pct = pct / 100
        pct_positive = copy.deepcopy(pct)
        pct_positive[pct_positive != np.nan] = 0
        pct_positive[pct > 0] = 1
        numerator = pct_positive.rolling(20).sum()
        PSY = (numerator / 20) * 100
        PSY = PSY.T
        PSY = CALFUNC.del_dat_early_than(PSY, START_YEAR)
        return PSY

    @lazyproperty
    def bias(self):
        close = self.closeprice_daily.T
        close.index = pd.to_datetime(close.index)
        MA20 = close.rolling(20, min_periods=1).mean()
        numerator = close-MA20
        BIAS = numerator/MA20*100
        BIAS = BIAS.T
        BIAS = CALFUNC.del_dat_early_than(BIAS, START_YEAR)
        return BIAS

    @lazyproperty
    def macd_diff_dea(self):
        close = self.closeprice_daily
        adj = self.adjfactor
        close, adj = align(close, adj)
        c_p = close.mul(adj)

        def EMA(arr, period=21):
            df = pd.DataFrame(arr)
            tt = df.ewm(span=period, min_periods=period, axis=1, ignore_na=True).mean()
            return tt

        def myMACD(close, fastperiod=10, slowperiod=30, signalperiod=15):
            ewma10 = EMA(close, fastperiod)
            ewma30 = EMA(close, slowperiod)
            dif = ewma10 - ewma30
            dea = EMA(dif, signalperiod)
            bar = (dif - dea) * 2
            return dif, dea, bar

        diff, dea, macd = myMACD(c_p)
        diff = CALFUNC.del_dat_early_than(diff, START_YEAR)
        dea = CALFUNC.del_dat_early_than(dea, START_YEAR)
        macd = CALFUNC.del_dat_early_than(macd, START_YEAR)

        res_dict = {"DIF": diff, "DEA": dea, "MACD": macd}

        return res_dict

    # 高管薪酬前三的合
    @lazyproperty
    def mgmt_ben_top3m(self):
        mgmt_ben_top3m = self.stmnote_mgmt_ben_top3m
        mgmt_ben_top3m = np.log(mgmt_ben_top3m)
        mgmt_ben_top3m = adjust_months(mgmt_ben_top3m, orig='Y')
        mgmt_ben_top3m = append_df(mgmt_ben_top3m)

        return mgmt_ben_top3m


if __name__ == "__main__":
    0
