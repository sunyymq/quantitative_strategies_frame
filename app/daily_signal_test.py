# 每日运行，检查出现的突破信号。
import numpy as np
import pandas as pd
import os
from datetime import datetime
from utility.tool0 import Data
from utility.constant import data_dair, root_dair

from app.data_management_class import Data_Management
from factor_compute_and_update.factor_compute import Factor_Compute


# 技术突破信号
def signal_test(cl, op, rps):

    # RPS是否大于85
    if rps[-1] < 85:
        res = False
    else:
        # 是否突破过去N天的最高价
        n = 50
        highest = cl[cl.index[-n-1:-1]].max()
        if cl[-1] > highest:
            res = True
        else:
            res = False

    return res


# main
def daily_test(date_para=None):

    # 更新价格数据
    data_manage = Data_Management()
    data_manage.update_market_quote_daily()

    # 更新RPS数据
    fc = Factor_Compute('update')  # ('update')
    res = fc.Rps
    fc.save(res, 'Rps'.upper())

    data = Data()
    basic = data.stock_basic_inform
    rps = data.RPS

    close = data.closeprice_daily
    op = data.openprice_daily
    high = data.highprice_daily
    low = data.lowprice_daily
    adj = data.adjfactor

    close_adj = close * adj
    open_adj = op * adj
    # high_adj = high * adj
    # low_adj = low * adj

    negotiablemv = data.negotiablemv_daily
    negotiablemv = negotiablemv[negotiablemv.columns[-1]]

    mv_tmp = negotiablemv.sort_values(ascending=False)
    mv_tmp = mv_tmp.dropna()
    qua = int(len(mv_tmp.index) * 0.5)

    big_size = mv_tmp.index[:qua]

    satisfied_list = []
    for code in basic.index:
        if code not in big_size:
            continue

        if not (code in close_adj.index and code in rps.index):
            continue

        # 不指定交易日，即最新的交易日
        if not date_para:
            close_se = close_adj.loc[code, :]
            open_se = open_adj.loc[code, :]
            rps_se = rps.loc[code, :]
        else:
            close_se = close_adj.loc[code, date_para]
            open_se = open_adj.loc[code, date_para]
            rps_se = rps.loc[code, date_para]

        if len(close_se.dropna()) < 50:
            continue

        close_se = close_se.dropna()
        open_se = open_se.dropna()

        # 检查是否满足突破条件
        tof = signal_test(cl=close_se, op=open_se, rps=rps_se)

        if tof:
            satisfied_list.append(code)
        # except Exception as e:
        #     print('deubg')

    return basic.loc[satisfied_list, 'SEC_NAME']


if "__main__" == __name__:
    res = daily_test()
    print(res)
    res.to_csv(r'D:\Database_Stock\股票池_最终\每日股票信号跟踪.csv', encoding='gbk')






