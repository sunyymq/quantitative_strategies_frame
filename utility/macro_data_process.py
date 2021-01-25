# 板块选择
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from datetime import datetime
from utility.factor_data_preprocess import adjust_months, add_to_panels, align, append_df
from utility.stock_pool import naturedate_to_tradeddate, fill_na_by_proceed
from utility.tool0 import Data, ma
from utility.relate_to_tushare import stocks_basis, generate_months_ends


'''
关于宏观数据的下载，在 utility.download_from_wind 中的 update_macro_data 函数中。

本函数库包含处理宏观数据部分的函数，包括处理日度宏观数据的deal_daily_macro_data，处理月度宏观数据的deal_month_macro_data，
给宏观数据做均值化处理的macro_data_smooth_process，以及同时调用上述三个函数，并返回一个结果的deal_marco_data。
进一步的改进就是在函数中添加宏观变量名称的参数。
'''





def deal_daily_macro_data(daily_dat):
    if '频率' in daily_dat.index:
        daily_dat.drop('频率', axis=0, inplace=True)

    # 先移动
    daily_macro_data = daily_dat.shift(1)
    # 再删除
    daily_macro_data.dropna(how='all', inplace=True)

    # 添加NAN
    for i in range(1, len(daily_macro_data.index)):
        for j in range(0, len(daily_macro_data.columns)):
            if pd.isna(daily_macro_data.iloc[i, j]) and not pd.isna(daily_macro_data.iloc[i-1, j]):
                daily_macro_data.iloc[i, j] = daily_macro_data.iloc[i - 1, j]

    month_ends = generate_months_ends()

    selected_index = [m for m in month_ends if m in daily_macro_data.index]
    res_df = daily_macro_data.loc[selected_index, :]
    res_df['利差'] = res_df['中债企业债到期收益率（AAA）：1年'] - res_df['中债国债到期收益率：1年']
    res_df.drop('中债企业债到期收益率（AAA）：1年', axis=1, inplace=True)

    return res_df


def deal_month_macro_data(month_dat):

    shift_month_num = {0: ['短期贷款利率:6个月至1年', '中长期贷款利率:1至3年', 'PMI', 'PMI:产成品库存'],
                       1: ['金融机构:人民币:资金运用合计', 'M1:同比', 'M2:同比', '社会融资规模:当月值',
                           'CPI当月同比', '出口金融:当月同比', '国房景气指数', '商品房销售面积：累计同比',
                           '房屋施工面积：累计同比', '房屋竣工面积：累计同比', '房地产开发投资完成额：累计同比',
                           'PPI:全部工业品：当月同比', 'PPI:建筑材料工业：当月同比', 'PPI:机械工业：当月同比',
                           '固定资产投资完成额：累计同比', '新增固定资产投资完成额：累计同比',
                           '固定资产投资完成额：基础设施建设投资：累计同比'
                           ]
                       }

    # 要计算同比的宏观指标
    to_yoy = ['金融机构:人民币:资金运用合计', '社会融资规模:当月值']

    # 处理宏观数据移动
    shifted_macro = pd.DataFrame()
    for col, se in month_dat.iteritems():
        finded = None
        for k, v in shift_month_num.items():
            if col in v:
                finded = k
                break

        if np.isnan(finded):
            print('{},该宏观数据未定义滞后期，错误'.format(col))
            raise KeyError

        tmp_pd = pd.DataFrame({col: se.shift(finded)})
        shifted_macro = pd.concat([shifted_macro, tmp_pd], axis=1)

        # 计算同比指标
        if col in to_yoy:
            # 计算同比
            tmp_pd = tmp_pd.diff(12)/tmp_pd.shift(12)
            tmp_pd.columns = [col+'_同比']
            shifted_macro = pd.concat([shifted_macro, tmp_pd], axis=1)

    shifted_macro.index.name = 'Date'
    return shifted_macro


def macro_data_smooth_process(macro_dat, para_dict=None, retain_num=6):

    # 对Nan进行填充，因为求均值，如果不处理nan的话，一个nan就会变成m个nan.

    macro_dat_v = np.array(macro_dat)
    [h, l] = macro_dat_v.shape
    for hh in range(1, h):
        for ll in range(0, l):
            if np.isnan(macro_dat_v[hh, ll]) and not np.isnan(macro_dat_v[hh-1, ll]):
                macro_dat_v[hh, ll] = macro_dat_v[hh - 1, ll]

    new_macro_dat = pd.DataFrame(data=macro_dat_v, index=macro_dat.index, columns=macro_dat.columns)

    if not para_dict:
        # 对宏观数据进行平滑处理
        para_dict = {'PMI': 3,
                     'CPI': 12,
                     'M': 6,
                     '汇率': 12,
                     '国债': 6,
                     '利差': 6,
                     }
    dat_smooth_ed = pd.DataFrame()
    for ind, se in new_macro_dat.iteritems():
        para_tmp = retain_num
        for k, v in para_dict.items():
            if k in ind:
                para_tmp = v
                break

        try:
            tt_df = ma(pd.DataFrame({ind: se}), para_tmp)
        except Exception as e:
            print('debug')
        dat_smooth_ed = pd.concat([dat_smooth_ed, tt_df], axis=1)

    dat_smooth_ed.dropna(how='all', inplace=True)
    dat_smooth_ed.index.name = 'date'

    return dat_smooth_ed


def deal_marco_data():
    data = Data()
    month_macro_data = data.month_macro_data_raw
    daily_macro_data = data.daily_macro_data_raw

    d_m = deal_daily_macro_data(daily_macro_data.T)
    dm0 = naturedate_to_tradeddate(d_m, tar='index')
    m_m = deal_month_macro_data(month_macro_data.T)
    dm1 = naturedate_to_tradeddate(m_m, tar='index')
    macro_dat = pd.concat([dm0, dm1], axis=1)
    smoothed_macro = macro_data_smooth_process(macro_dat)
    return smoothed_macro


