import pandas as pd
import numpy as np
from datetime import datetime
import os
from time import sleep
import copy
import os
from datetime import datetime, timedelta
import tushare as ts
import shelve
from utility.tool0 import Data
from utility.constant import NH_index_dict, tds_interface, data_dair
from utility.relate_to_tushare import generate_months_ends
from WindPy import *

if tds_interface == 'tushare':
    token_path = r'C:\token_tushare.txt'
    if os.path.exists(token_path):
        f = open(token_path)
        token = f.read()
        f.close()

        ts.set_token(token)
        pro = ts.pro_api()

pro = ts.pro_api()

# # 基金列表
# df_e = pro.fund_basic(market='E')
# df_o = pro.fund_basic(market='O')

# 单只基金
df0 = pro.fund_share(ts_code='150018.SZ')
df1 = pro.fund_share(ts_code='150018.SZ', start_date='20120806', end_date='20190806')


def get_fund_basic():

    basic_path = r'D:\Database_Stock\Data\fund\basic'
    mix_fund = pd.read_csv(os.path.join(basic_path, 'mix_fund.CSV'))
    stock_fund = pd.read_csv(os.path.join(basic_path, 'stock_fund.CSV'))

    # 两个合并，然后删除基金名称里有'量化'、'指数增强'字符的基金
    res = pd.concat([mix_fund, stock_fund], axis=0)
    res.index = range(0, len(res))
    to_del = []
    for i, se in res.iterrows():
        if '量化' in se['证券简称']:
            to_del.append(i)
        elif '多因子' in se['证券简称']:
            to_del.append(i)

    for i, se in res.iterrows():
        if se['证券简称'].endswith('A'):
            se['证券简称'] = se['证券简称'][:-1]
        elif se['证券简称'].endswith('B'):
            se['证券简称'] = se['证券简称'][:-1]
        elif se['证券简称'].endswith('C'):
            se['证券简称'] = se['证券简称'][:-1]
        elif se['证券简称'].endswith('E'):
            se['证券简称'] = se['证券简称'][:-1]

    res.drop_duplicates(subset=['证券简称'], keep='first', inplace=True)

    return res


# 根据tushare里面的函数返回值，得该基金的截面的结果
def arrange_fun(dat_df):
    stk_ratio = pd.DataFrame()

    grouped = dat_df.groupby('end_date')
    for i, df_tmp in grouped:
        group_tmp = df_tmp.groupby('ann_date')
        ann_dict = {}
        for d, tmp in group_tmp:
            ann_dict.update({d: tmp})

        if len(ann_dict.keys()) > 1:
            # 找到披露最早的日期
            dates = list(ann_dict.keys())
            shift_days = [(dd - pd.to_datetime([i])).days for dd in pd.to_datetime(dates)]
            loc = shift_days.index(np.min(shift_days))

            ann_date = dates[loc]
            i_df = ann_dict[dates[loc]]
        else:
            ann_date = df_tmp['ann_date'].values[0]
            i_df = df_tmp

        # 数据库的问题，重复的个股
        if i_df['symbol'].duplicated().any():
            i_df = i_df.drop(i_df.index[i_df['symbol'].duplicated()], axis=0)

        if len(i_df) > 10:
            i_df = i_df.sort_values('mkv', ascending=False)
            i_df = i_df.iloc[0:10, :]

        # 保持重仓股的市值，index是所有股票代码
        i_df = i_df.set_index('symbol')
        try:
            stk_ratio = pd.concat([stk_ratio, pd.DataFrame({ann_date: i_df['stk_float_ratio']})], axis=1)
        except Exception as e:
            print('dddd')

    stk_ratio.columns = pd.to_datetime(stk_ratio.columns)

    return stk_ratio


# 下载基金季度重仓股数据，并保持到db数据库中
def download_top10_portfolio():
    fund_b = get_fund_basic()

    float_rate_dict = {}
    for fund_code in fund_b['证券代码']:
        df_tmp = pro.fund_portfolio(ts_code=fund_code)
        stk_rate_df = arrange_fun(df_tmp)
        float_rate_dict.update({fund_code: stk_rate_df})
        sleep(1)

    # 保存原始数据
    db = shelve.open(r'D:\Database_Stock\Data\fund\fund_portfolio.db')
    db['fund_portfolio_float_rate'] = float_rate_dict
    db.close()

    return 0


# 下载基金复权净值数据
def fund_nav():
    fund_b = get_fund_basic()

    res_df = pd.DataFrame()
    for code in fund_b['证券代码']:
        sleep(1)
        print(code)
        tmp = pro.fund_nav(ts_code=code)
        tmp_df = tmp[['ann_date', 'adj_nav']]
        tmp_df = tmp_df.set_index('ann_date')
        tmp_df.index = pd.to_datetime(tmp_df.index)
        tmp_df.sort_index(inplace=True)
        tmp_df.columns = [code]
        tmp_df = tmp_df.loc[tmp_df.index.drop_duplicates(keep=False), :]

        # tmp_df.index.duplicated()
        # np.sum(pd.isna(tmp_df))

        try:
            res_df = pd.concat([res_df, tmp_df], axis=1, join='outer')
        except Exception as e:
            print('debug')

    return res_df


# 根据每个基金的重仓股数据，整理出index为股票代码，columns为日期的截面形式。
def reorganize_fund_code_dat(fund_holding_dict):

    mes = generate_months_ends()
    data = Data()
    stock_basic_inform = data.stock_basic_inform

    res_df = pd.DataFrame(0, index=stock_basic_inform.index, columns=mes)
    for key, hold_df in fund_holding_dict.items():

        for col in hold_df.columns:
            # 选择出对应的月份
            finded = 0
            for all_col in res_df.columns:
                if all_col.year == col.year and all_col.month == col.month:
                    finded = all_col
                    break

            # 太早的数据，如02年的
            if finded == 0:
                continue

            if col.month not in [1, 4, 7, 10]:
                continue

            try:
                target_month = res_df[finded]
            except Exception as e:
                print('debug')

            # 针对对应的股票，数量加1
            for stock in hold_df[col]:
                if stock in target_month.index:
                    target_month[stock] = target_month[stock] + 1

    # 对于空余的月份，直接复制前值
    for col_n in range(1, len(res_df.columns)):
        if res_df[res_df.columns[col_n]].sum() == 0 and res_df[res_df.columns[col_n-1]].sum() != 0:
            res_df[res_df.columns[col_n]] = res_df[res_df.columns[col_n-1]]

    tt = res_df[[res_df.columns[-1]]]
    tt = tt.sort_values(by=tt.columns[0], ascending=False)
    tt['rank'] = range(1, len(tt)+1)

    return res_df


# 根据每个基金的重仓股持股市值数据，整理出index为股票代码，columns为日期的截面形式。
def reorganize_fund_mkv_dat(fund_holding_mkv_dict):
    mes = generate_months_ends()
    data = Data()
    stock_basic_inform = data.stock_basic_inform

    res_df = pd.DataFrame(0, index=stock_basic_inform.index, columns=mes)
    for key, hold_df in fund_holding_mkv_dict.items():

        for col in hold_df.columns:
            # 选择出对应的月份
            finded = 0
            for all_col in res_df.columns:
                if all_col.year == col.year and all_col.month == col.month:
                    finded = all_col
                    break

            # 太早的数据，如02年的
            if finded == 0:
                continue

            # 一个小bug
            if col.month not in [1, 4, 7, 10]:
                continue

            try:
                target_month = res_df[finded]
            except Exception as e:
                print('debug')

            tmp_se = hold_df[col].dropna()
            # 针对对应的股票，数量加1
            for stock in tmp_se.index:
                if stock in target_month.index:
                    target_month[stock] = target_month[stock] + tmp_se[stock]

    test = (res_df > 0).sum()

    # 对于空余的月份，直接复制前值
    for col_n in range(1, len(res_df.columns)):
        if res_df[res_df.columns[col_n]].sum() == 0 and res_df[res_df.columns[col_n - 1]].sum() != 0:
            res_df[res_df.columns[col_n]] = res_df[res_df.columns[col_n - 1]]

    return res_df

    # nav = fund_nav()
    # basic_path = r'D:\Database_Stock\Data\fund'
    # nav.to_csv(os.path.join(basic_path, 'adj_nav.csv'), encoding='gbk')


# 更新基金重仓股数据
def update_fund_dat(mode='update'):

    # 仅更新因子数据
    if mode == 'update':

        # 伪代码：先读取历史的因子数据，检查是否需要更新，如无需更新就直接退出
        if 0:
            pass
        else:
            print('该因子无需更新，退出')
            return

        # 伪代码： 如果月份是1、4、7、10月，重新计算该数据
        if datetime.today().month in [1, 4, 7, 10]:
            download_top10_portfolio()

    elif mode == 'renew':
        download_top10_portfolio()


        #     # 保存原始数据
        #     db = shelve.open(r'D:\Database_Stock\Data\fund\fund_portfolio.db')
        #     db['fund_portfolio_code'] = res_code_dict
        #     db['fund_portfolio_MKV'] = res_mkv_dict
        #     db.close()
        #
        #     # 读取原始数据以备它用
        #     # db = shelve.open(r'D:\Database_Stock\Data\fund\fund_portfolio.db')
        #     # res_code_dict = db['fund_portfolio_code']
        #     # res_mkv_dict = db['fund_portfolio_MKV']
        #     # db.close()
        #
        #     # 重仓持有该个股的基金的数量
        #     holding_fund_num = reorganize_fund_code_dat(res_code_dict)
        #     # 重仓持有该个股的市值和
        #     mkv_fund_stock = reorganize_fund_mkv_dat(res_mkv_dict)
        #
        #     # 保持的地址
        #     p = r'D:\Database_Stock\Data\factor_data'
        #
        #     # 保存为因子截面数据
        #     mkv_fund_stock.to_csv(os.path.join(p, 'fund_holding_mkv'.upper() + '.csv'))
        #     holding_fund_num.to_csv(os.path.join(p, 'fund_holding_num'.upper() + '.csv'))
        #
        # # 伪代码： 如果月份是其他月份，基金持仓数据未更新，读取历史的数据，直接使用前一个月的数据
        # else:
        #     mes = generate_months_ends()
        #     to_update_month = [m for m in mes if m not in mkv_fund_stock.columns and m > mkv_fund_stock.columns[0]]


if '__main__' == __name__:
    # 下载基金复权净值数据
    # fund_adj_nav = fund_nav()
    # p = r'D:\Database_Stock\Data\fund'
    # # 保存到本地
    # fund_adj_nav.to_csv(os.path.join(p, 'fund_adj_nav' + '.csv'))

    update_fund_dat(mode='renew')


    # # 读取原始数据以备它用
    # db = shelve.open(r'D:\Database_Stock\Data\fund\fund_portfolio.db')
    # res_code_dict = db['fund_portfolio_code']
    # res_mkv_dict = db['fund_portfolio_MKV']
    # db.close()
    #
    # # 重仓持有该个股的基金的数量
    # holding_fund_num = reorganize_fund_code_dat(res_code_dict)
    # # 重仓持有该个股的市值和
    # mkv_fund_stock = reorganize_fund_mkv_dat(res_mkv_dict)
    #
    # # 保持的地址
    # p = r'D:\Database_Stock\Data\factor_data'
    #
    # # 保存为因子截面数据
    # mkv_fund_stock.to_csv(os.path.join(p, 'fund_holding_mkv'.upper() + '.csv'))
    # holding_fund_num.to_csv(os.path.join(p, 'fund_holding_num'.upper() + '.csv'))



