# 每日运行，检查最近的强势股。
import numpy as np
import pandas as pd
import os
from utility.tool0 import Data
from utility.constant import data_dair, root_dair


def daily_select():
    data = Data()
    basic = data.stock_basic_inform
    rps = data.RPS
    rps_by_industry = data.RPS_BY_INDUSTRY

    # 前期的股票池
    pre_pool_path = os.path.join(root_dair, '临时', 'double_rps_pool.csv')
    if os.path.exists(pre_pool_path):
        pre_pool = pd.read_csv(pre_pool_path, encoding='gbk')
        pre_pool.set_index(pre_pool.columns[0], inplace=True)
    else:
        pre_pool = None

    # 部分分析过的基本面较差的股票池，如果在的话，可以删掉
    to_drop_path = os.path.join(root_dair, '临时', 'to_drop_code.csv')
    if os.path.exists(to_drop_path):
        to_drop = pd.read_csv(to_drop_path)
        to_drop.set_index(to_drop.columns[0], inplace=True)
    else:
        to_drop = None

    rps_se = rps[rps.columns[-1]]
    rps_by_industry_se = rps_by_industry[rps_by_industry.columns[-1]]

    rps_over_80 = rps_se[rps_se > 80]
    rps_by_ind_over_80 = rps_by_industry_se[rps_by_industry_se > 80]
    double_rps_index = list(set(rps_over_80.index) & set(rps_by_ind_over_80.index))
    rps_mean = (rps_over_80[double_rps_index] + rps_by_ind_over_80[double_rps_index])/2

    last = os.listdir(os.path.join(root_dair, '因子预处理模块', '因子'))[-1]
    last_p = os.path.join(root_dair, '因子预处理模块', '因子', last)
    section_df = pd.read_csv(last_p, encoding='gbk')
    section_df.set_index('Code', inplace=True)

    section = section_df.loc[double_rps_index, ['Roe_q', 'Roa_q', 'Profit_g_q', 'Roe_g_q', 'Sue', 'West_netprofit_yoy']]

    dr = basic.loc[double_rps_index, ['SEC_NAME', '申万一级行业', '申万三级行业']]
    dr = pd.concat([dr, pd.DataFrame({'RPS_MEAN': rps_mean}), section], axis=1)
    dr = dr.dropna(how='any', axis=0)
    dr = dr.sort_index()

    if isinstance(to_drop, pd.DataFrame):
        tmp = [i for i in dr.index if i not in to_drop.index]
        dr = dr.loc[tmp, :]

    if isinstance(pre_pool, pd.DataFrame):

        get_in = [i for i in dr.index if i not in pre_pool.index]
        throw_out = [i for i in pre_pool.index if i not in dr.index]

        dr.to_csv(pre_pool_path, encoding='gbk')
        if len(get_in) > 0:
            in_df = dr.loc[get_in, :]
            p = os.path.join(root_dair, '临时', 'double_rps_pool_新调入.csv')
            in_df.to_csv(p, encoding='gbk')
        else:
            in_df = None

        if len(throw_out) > 0:
            out_df = pre_pool.loc[throw_out, :]
            p = os.path.join(root_dair, '临时', 'double_rps_pool_被调仓.csv')
            out_df.to_csv(p, encoding='gbk')
        else:
            out_df = None

    else:
        print('无前置股票池，直接保存现有股票池')
        dr.to_csv(pre_pool_path, encoding='gbk')
        in_df = None
        out_df = None

    return dr, in_df, out_df


if __name__ == "__main__":
    dr, in_df, out_df = daily_select()


