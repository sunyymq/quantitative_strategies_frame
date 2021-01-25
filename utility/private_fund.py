import os
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from openpyxl import load_workbook
import pandas.tseries.offsets as toffsets
from utility.analysis import Analysis

benchmark_map = {0: 'HS300',
                 1: 'Wind股票策略私募基金指数',
                 2: 'Wind股票市场中性私募基金指数',
                 3: 'Wind套利策略私募基金指数',
                 4: 'Wind多策略私募基金指数',
                 5: 'Wind债券策略私募基金指数',
                 6: 'Wind组合基金策略私募基金指数',
                 7: 'ZZ500'
                }

strategy_index_path = r'D:\私募基金相关\私募基金策略指数'


def load_bench_dat(bench_type):
    if bench_type not in benchmark_map.keys():
        raise KeyError
    v = benchmark_map[bench_type]
    p = os.path.join(strategy_index_path, v + '.xlsx')
    bench = pd.read_excel(p, encoding='gbk')
    bench.set_index('日期', inplace=True)

    return bench


if __name__ == '__main__':
    path = r'D:\Database_Stock\历史净值.xlsx'
    dat_df = pd.read_excel(path)
    dat_df = dat_df.set_index('日期')
    dat_df.index = pd.to_datetime(dat_df.index)
    dat_df = dat_df.sort_index()

    bench = load_bench_dat(1)

    private_equity = Analysis(dat_df, 'w', bench, rf_rate=0.04)
    summary = private_equity.summary()
    save_path = r'D:\Database_Stock\评价.xlsx'
    summary.to_excel(save_path, encoding='gbk')


