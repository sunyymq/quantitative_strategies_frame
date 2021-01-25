# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:47:11 2019

@author: admin
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from itertools import chain
from functools import reduce
from sklearn.linear_model import LinearRegression
from utility.constant import info_cols, data_dair, non_processed_factors
from utility.relate_to_tushare import trade_days
from utility.tool0 import Data
import statsmodels.api as sm
from pyfinance.utils import rolling_windows


def align(df1, df2, *dfs):
    # chain 是把多个迭代器合成一个迭代器
    dfs_all = [df for df in chain([df1, df2], dfs)]
    # 看df1和df2是否有单个列的
    if any(len(df.shape) == 1 or 1 in df.shape for df in dfs_all):
        dims = 1
    else:
        dims = 2
    # 对日期求交期. reduce: 用传给reduce中的函数function（有两个参数）先对集合中的第 1、2个元素进行操作，
    # 得到的结果再与第三个数据用function函数运算，最后得到一个结果。
    mut_date_range = sorted(reduce(lambda x, y: x.intersection(y), (df.index for df in dfs_all)))
    # 对columns求交集
    mut_codes = sorted(reduce(lambda x, y: x.intersection(y), (df.columns for df in dfs_all)))
    # 如果df1和df2都是多维的，求日期和代码的交集；否则，只求日期的交集
    if dims == 2:
        dfs_all = [df.loc[mut_date_range, mut_codes] for df in dfs_all]
    elif dims == 1:
        dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
    return dfs_all


def drop_some(datdf):
    global info_cols

    cond = pd.Series(True, index=datdf.index)
    # 最新一期数据
    if pd.isnull(datdf['Pct_chg_nm']).all():
        pass
    else:
        # 删除未上市股票
        cond &= ~pd.isnull(datdf['Mkt_cap_float'])
        # 删除未开盘股票
        cond &= datdf['Is_open1']

    datdf = datdf.loc[cond]

    return datdf


def fill_na(data, ind='sw', fill_type='any'):
    """
    缺失值填充：缺失值少于10%的情况下使用行业中位数代替
    """
    global info_cols, non_processed_factors
    datdf = data.copy()
    if ind == 'sw':
        datdf = datdf.loc[~pd.isnull(datdf['Industry_sw']), :]

    non_deal = info_cols + non_processed_factors
    tmp_info_cols = [inf for inf in non_deal if inf in datdf.columns]
    # datdf中剔除info_cols后的列名
    facs_to_fill = datdf.columns.difference(set(tmp_info_cols))

    datdf[facs_to_fill] = datdf[facs_to_fill].applymap(coerce_numeric)
    datdf = datdf.replace([np.inf, -np.inf], np.nan)    # 替换inf

    # pd.to_numeric( datdf[facs_to_fill], errors='coerce')
    if fill_type != 'any':
        facs_to_fill = [fac for fac in facs_to_fill            # 筛选缺失值少于10%的因子
                            if pd.isnull(datdf[fac]).sum() / len(datdf) <= 0.1]
    else:
        facs_to_fill = [fac for fac in facs_to_fill            # 筛选缺失值少于10%的因子
                        if pd.isnull(datdf[fac]).any()]

    if ind in ['zx', 'sw']:
        grouped_column = f'Industry_{ind}'
    elif ind == 'Second_industry':
        grouped_column = 'Second_industry'
    else:
        raise Exception

    for fac in facs_to_fill:
        fac_median_by_ind = datdf[[grouped_column, fac]].groupby(grouped_column).median()
        # 把dateframe转为dict,并取fac为key以解决 dict套dict 的问题
        fac_ind_map = fac_median_by_ind.to_dict()[fac]
        # 选出需要替换的数据
        fac_to_fill = datdf.loc[pd.isnull(datdf[fac]), [grouped_column, fac]]
        # map函数可以接受含有映射关系的字典。使用map做行业到其均值的映射。
        fac_to_fill.loc[:, fac] = fac_to_fill[grouped_column].map(fac_ind_map)
        # 添加回到datdf
        datdf.loc[fac_to_fill.index, fac] = fac_to_fill[fac].values
        if pd.isnull(datdf[fac]).any():
            datdf[fac] = datdf[fac].fillna(np.nanmean(datdf[fac]))

    # 针对sw行业存在缺失值的情况
    if len(datdf) < len(data):
        idx_to_append = data.index.difference(datdf.index)
        datdf = pd.concat([datdf, data.loc[idx_to_append, :]])
        datdf.sort_index()

    return datdf


def coerce_numeric(s):
    try:
        return float(s)
    except:
        return np.nan


def winsorize(data, n=5):
    """
    去极值：5倍中位数标准差法（5mad）
    """
    global info_cols, non_processed_factors
    
    datdf = data.copy()
    non_deal = info_cols + non_processed_factors
    tmp_info_cols = [inf for inf in non_deal if inf in datdf.columns]

    # 找出含有 nan 的列
    if_contain_na = pd.isnull(datdf).sum().sort_values(ascending=True)
    facs_to_remove = if_contain_na.loc[if_contain_na > 0].index.tolist()
    if 'PCT_CHG_NM' in facs_to_remove:
        facs_to_remove.remove('PCT_CHG_NM')

    # 剔除含有 nan 的列 和 info_cols的列 后的所有列
    facs_to_win = datdf.columns.difference(set(tmp_info_cols)).difference(set(tuple(facs_to_remove)))
    dat_win = datdf[facs_to_win]
    dat_win = dat_win.applymap(apply_func2)
    fac_vals = dat_win.values

    # np.median(fac_vals)
    try:
        dm = np.nanmedian(fac_vals, axis=0)
    except Exception as e:
        print('debug')
    # 与均值差的绝对值的非 nan 均值
    dm1 = np.nanmedian(np.abs(fac_vals - dm), axis=0)
    if 0 in (dm + n*dm1): 
        # 针对存在去极值后均变为零的特殊情况（2009-05-27-'DP')
        cut_points = [i for i in np.argwhere(dm1 == 0)[0]]
        # 提取对应列，对其不进行去极值处理
        facs_unchanged = [facs_to_win[cut_points[i]] for i in range(len(cut_points))] 
        # 仅对剩余列进行去极值处理
        facs_to_win_median = facs_to_win.difference(set(tuple(facs_unchanged)))
        
        dat_win_median = datdf[facs_to_win_median]

        def fun1(x):
            try:
                r = float(x)
            except Exception as e:
                r = 0
            return r
        dat_win_median = dat_win_median.applymap(fun1)

        fac_median_vals = dat_win_median.values
        dmed = np.nanmedian(fac_median_vals, axis=0)
        dmed1 = np.nanmedian(np.abs(fac_median_vals - dmed), axis=0)
        dmed = np.repeat(dmed.reshape(1,-1), fac_median_vals.shape[0], axis=0)
        dmed1 = np.repeat(dmed1.reshape(1,-1), fac_median_vals.shape[0], axis=0)
        
        fac_median_vals = np.where(fac_median_vals > dmed + n*dmed1, dmed+n*dmed1, 
              np.where(fac_median_vals < dmed - n*dmed1, dmed - n*dmed1, fac_median_vals))
        res1 = pd.DataFrame(fac_median_vals, index=dat_win_median.index, columns=dat_win_median.columns)
        res2 = datdf[facs_unchanged]
        res = pd.concat([res1, res2], axis=1)
    else:
        # 通过两个repeat，得到与fac_vals 中元素一一对应的极值
        dm = np.repeat(dm.reshape(1, -1), fac_vals.shape[0], axis=0)
        dm1 = np.repeat(dm1.reshape(1, -1), fac_vals.shape[0], axis=0)
        # 替换
        fac_vals = np.where(fac_vals > dm + n*dm1, dm+n*dm1, 
              np.where(fac_vals < dm - n*dm1, dm - n*dm1, fac_vals))
        res = pd.DataFrame(fac_vals, index=dat_win.index, columns=dat_win.columns)

    datdf[facs_to_win] = res
    return datdf  


def neutralize(data, ind_neu=True, size_neu=True, ind='sw', plate=None):
    """
    中性化：因子暴露度对行业哑变量（ind_dummy_matrix）和对数流通市值（lncap_barra）
            做线性回归, 取残差作为新的因子暴露度
    """
    global info_cols, non_processed_factors
    datdf = data.copy()
    if ind == 'sw':
        datdf = datdf.loc[~pd.isnull(datdf['Industry_sw']), :]

    non_deal = info_cols + non_processed_factors
    tmp_info_cols = [inf for inf in non_deal if inf in datdf.columns]

    # 剔除 info_cols 这些列后剩下的列名
    cols_to_neu = datdf.columns.difference(set(tmp_info_cols))
    y = datdf[cols_to_neu]
    # 剔除含有nan的
    y = y.dropna(how='any', axis=1)
    cols_neu = y.columns

    if size_neu:
        # 对数市值
        lncap = np.log(datdf[['Mkt_cap_float']])

    # 若针对特定行业，则无需生成行业哑变量
    use_dummies = 1

    if not ind_neu:
        use_dummies = 0

    # 市值中性行业不中性
    if use_dummies == 0 and size_neu:
        X = lncap
    # 行业中性市值不中性
    elif use_dummies == 1 and not size_neu:
        X = pd.get_dummies(datdf[f'Industry_{ind}'])
    else:
        # 使用 pd.get_dummies 生成行业哑变量
        ind_dummy_matrix = pd.get_dummies(datdf[f'Industry_{ind}'])
        # 合并对数市值和行业哑变量
        X = pd.concat([lncap, ind_dummy_matrix], axis=1)

    model = LinearRegression(fit_intercept=False)
    # 一次对所有的y都做回归
    try:
        res = model.fit(X, y)
    except Exception as e:
        pd.isna(y).sum().sum()
        pd.isna(X).sum().sum()
        for col, se in y.iteritems():
            pd.isna(se).sum()
            (se == -np.inf).sum()
            np.where(se == -np.inf)
            np.where(se == np.inf)
            print(col)
            res = model.fit(X, se)
            print('debug')
    coef = res.coef_
    residue = y - np.dot(X, coef.T)

    # 断言语言， 如果为false则触发错误
    assert len(datdf.index.difference(residue.index)) == 0

    datdf.loc[residue.index, cols_neu] = residue
    return datdf


def standardize(data):
    """
    标准化：Z-score标准化方法，减去均值，除以标准差
    """
    global info_cols, non_processed_factors
    
    datdf = data.copy()

    non_deal = info_cols + non_processed_factors
    tmp_info_cols = [inf for inf in non_deal if inf in datdf.columns]

    facs_to_sta = datdf.columns.difference(set(tmp_info_cols))
    
    dat_sta = np.float64(datdf[facs_to_sta].values)
    dat_sta = (dat_sta - np.nanmean(dat_sta, axis=0)) / np.nanstd(dat_sta, axis=0)

    datdf.loc[:, facs_to_sta] = dat_sta
    return datdf


def process_input_names(factor_names):
    if factor_names == 'a':
        factor_names = None
    else:
        factor_names = [f.replace("'", "").replace('"', "") for f in factor_names.split(',')]
    return factor_names


# 向现有的月度因子数据中添加一列因子
def add_columns(added_date_path, columns_list, target_date_path):
    '''
    :param added_date_path:     添加数据的存储位置
    :param columns_list:        准备添加的列名
    :param target_date_path:    需要被添加的数据存储位置
    :return:
    '''
    toadded_list = os.listdir(added_date_path)
    save_list = os.listdir(target_date_path)

    if pd.to_datetime(toadded_list[0].split('.')[0]) > pd.to_datetime(save_list[0].split('.')[0]) or \
            pd.to_datetime(toadded_list[-1].split('.')[0]) < pd.to_datetime(save_list[-1].split('.')[0]):
        print('被添加数据长度不够')
        raise Exception

    for panel_f in os.listdir(target_date_path):
        toadded_dat = pd.read_csv(os.path.join(added_date_path, panel_f),
                                  encoding='gbk', engine='python',
                                  index_col=['code'])

        panel_dat = pd.read_csv(os.path.join(target_date_path, panel_f),
                                encoding='gbk', engine='python',
                                index_col=['code'])

        real_add_list = [col for col in columns_list if col not in panel_dat.columns]
        if len(real_add_list) == 0:
            continue

        # join_axes关键字为沿用那个的index,忽略另一个df的其余数据
        panel_dat = pd.concat([panel_dat, toadded_dat[real_add_list]], axis=1, join_axes=[panel_dat.index])
        panel_dat.to_csv(os.path.join(target_date_path, panel_f),
                         encoding='gbk')

    print('数据添加完毕')


# 根据给定的日度日期序列和月末日期，找到该序列中该月末日期的月初日期
def getmonthfirstdate(dt, md):
    tmp1 = dt[dt.year == md.year]
    tmp2 = tmp1[tmp1.month == md.month]
    return tmp2[0]


# 得到给定日度时间序列的月末时间list
def get_monthends_series(dt):
    if isinstance(dt, pd.DataFrame):
        dt = list(dt)

    p = 0
    med = []
    for i in range(len(dt)-1):
        mon_t = dt[i].month
        mon_n = dt[i+1].month
        if mon_t != mon_n:
            med.append(dt[i])
            p = p + 1

    return pd.Series(med)


def simple_func(pd_s, mv, type='median'):
    # 市值加权
    if type == 'mv_weighted':
        tmpp = pd.concat([pd_s, mv], axis=1)
        tmpp = tmpp.dropna(axis=0)
        pd_s = tmpp[tmpp.columns[0]]
        mv = tmpp[tmpp.columns[1]]
        mv_weights = mv/np.sum(mv)
        v = np.dot(np.mat(pd_s), np.mat(mv_weights).T)
        return np.array(v).flatten()
    # 中位数
    elif type == 'median':
        return np.nanmedian(pd_s)
    elif type == 'mean':
        return np.nanmean(pd_s)
    else:
        raise Exception


def apply_func(df, mv, type='median'):
    # 市值加权
    if type == 'mv_weighted':
        mv_weights = mv/np.sum(mv)
        v = np.dot(np.mat(df), np.mat(mv_weights).T)
        return np.array(v).flatten()
    # 中位数
    elif type == 'median':
        return df.median()
    else:
        raise Exception


def apply_func2(x):
    if isinstance(x, str):
        try:
            x = float(x)
        except Exception as e:
            x = 0
    else:
        x
    return x


def concat_factor_2(data_path, save_path, classified_df, factor_name, wei_type, save_name):
    # 创建文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cols = set(list(classified_df[classified_df.columns[0]]))

    total_df = pd.DataFrame()

    for panel_f in os.listdir(data_path):
        print(panel_f)
        panel_dat = pd.read_csv(os.path.join(data_path, panel_f),
                                encoding='gbk', engine='python',
                                index_col=['code'])

        tmp_df = pd.concat([panel_dat[[factor_name, 'MKT_CAP_FLOAT']], classified_df], axis=1, join='inner')

        d = datetime.strptime(panel_f.split('.')[0], "%Y-%m-%d")
        section_df = pd.DataFrame(index=[d], columns=cols)

        grouped = tmp_df.groupby(classified_df.columns[0])
        for pla, group in grouped:
            group.dropna(how='any', inplace=True)
            section_df.loc[d, pla] = simple_func(group[factor_name], mv=group['MKT_CAP_FLOAT'], type='mv_weighted')[0]

        total_df = pd.concat([total_df, section_df], axis=0)

    if '.' not in save_name:
        save_name = save_name + '.csv'

    total_df.index.name = 'date'

    total_df.to_csv(os.path.join(save_path, save_name), encoding='gbk')
    # 做一个累计净值走势图
    # prod_total_df = (total_df + 1).cumprod()
    # prod_total_df.to_csv(os.path.join(save_path, '累计_'+save_name), encoding='gbk')


# 把一个 截面数据添加到已经有的月度模式存储的文件中
def add_to_panels(dat, panel_path, f_name, freq_in_dat='M'):
    """说明： 把dat依次插入到panel_path的DF中，插入的列名为f_name, 根据dat的类型是DF还是Series可以判断
    是每次插入的数据不同还是每次插入相同的数据。"""

    print(f'开始添加{f_name}数据到目标文件夹')
    panel = os.listdir(panel_path)
    for month_date in panel:
        hased_dat = pd.read_csv(os.path.join(panel_path, month_date), engine='python')
        hased_dat = hased_dat.set_index('Code')

        # 输入数据为 DataFrame, 那么按列插入
        if isinstance(dat, pd.DataFrame):
            mon_str = month_date.split('.')[0]
            if mon_str in dat.columns:
                # 当dat中的columns也是str格式，且日期与panel一样时，直接添加
                hased_dat[f_name] = dat[mon_str]
            else:
                # 否则，当年、月相同，日不同时，需要变成datetime格式而且还有查找
                target = datetime.strptime(mon_str, "%Y-%m-%d")
                # 当dat的columns是datetime格式时
                if isinstance(dat.columns[0], datetime):
                    if freq_in_dat == 'M':
                        finded = None
                        for col in dat.columns:
                            if col.year == target.year and col.month == target.month:
                                finded = col
                                break
                        if finded:
                            hased_dat[f_name] = dat[finded]
                        else:
                            print('{}该期未找到对应数据'.format(mon_str))
                    if freq_in_dat == 'D':
                        if target in dat.columns:
                            hased_dat[f_name] = dat[target]
                        else:
                            print('{}该期未找到对应数据'.format(mon_str))
                else:
                    print('现有格式的还未完善')
                    raise Exception
        # 输入数据为 DataFrame, 那么按列插入
        elif isinstance(dat, pd.Series):
            hased_dat[f_name] = dat[hased_dat.index]

        try:
            hased_dat = hased_dat.reset_index('Code')
        except Exception as e:
            print('debug')

        if 'No' in hased_dat.columns:
            del hased_dat['No']
        hased_dat.index.name = 'No'
        hased_dat.to_csv(os.path.join(panel_path, month_date), encoding='gbk')

    print('完毕！')


# 从一个月度panel里面删除某个因子
def del_factor_from_panel(panel_path, factor_name):

    print(f'开始从目标文件夹删除{factor_name}因子。')
    panel = os.listdir(panel_path)
    for month_date in panel:
        dat_df = pd.read_csv(os.path.join(panel_path, month_date), engine='python')
        dat_df = dat_df.set_index('Code')

        if factor_name in dat_df.columns:
            del dat_df[factor_name]
            dat_df.reset_index(inplace=True)
            dat_df.set_index('No', inplace=True)
            dat_df.index.name = 'No'
            dat_df.to_csv(os.path.join(panel_path, month_date), encoding='gbk')

    print(f'完毕。')
    return


def rolling_regress_1(y, x, window=5):
    try:
        rolling_ys = rolling_windows(y, window)
        rolling_xs = rolling_windows(x, window)
    except Exception as e:
        print('debug')

    bet = pd.Series()
    # enumerate 形成带 i 的一个迭代器
    for i, (rolling_x, rolling_y) in enumerate(zip(rolling_xs, rolling_ys)):
        tmp_index = y.index[i + window - 1]
        try:
            model = sm.OLS(rolling_y, rolling_x)
            result = model.fit()
            params = result.params
            b_v = params[0]
            # print(result.params)
            # print(result.summary())
        except:
            print(i)
            raise

        b = pd.Series(index=[tmp_index], data=b_v)
        bet = pd.concat([bet, b])

    return bet


# 计算不同股指合约的beta值
def compute_future_beta():
    # 存储地址为：D:\Datebase_Stock\Date\index\stock_future\sf_beta.csv
    data = Data()
    sf_close_daily = data.sf_close_daily
    index_price_daily = data.index_price_daily.T

    # 求一下日期的交集，避免日期不同的潜在问题
    tt = list(set(sf_close_daily.columns) & set(index_price_daily.index))
    tt.sort()

    sf_close_daily = sf_close_daily[tt]
    index_price_daily = index_price_daily.loc[tt, :]

    sf_beta = pd.DataFrame()
    for c, se in sf_close_daily.iterrows():
        if 'IC' in c:
            tmp_i = index_price_daily['ZZ500']
        elif 'IF' in c:
            tmp_i = index_price_daily['HS300']
        elif 'IH' in c:
            tmp_i = index_price_daily['SZ50']
        else:
            print('Code Bug')
            raise ValueError

        # 去掉Nan
        tmp_c = se.dropna()
        tmp_i = tmp_i[tmp_c.index]
        if len(tmp_c) > 22:
            bet = rolling_regress_1(tmp_i, tmp_c, window=22)
            sf_beta = pd.concat([sf_beta, pd.DataFrame({c: bet}).T], axis=0)

    p = os.path.join(data_dair, 'index', 'stock_future')
    data.save(sf_beta, 'sf_beta', p)


if __name__ == "__main__":

    # compute_future_beta()

    add_columns

    # panel_path = r"D:\pythoncode\IndexEnhancement\因子预处理模块\因子"
    # factor_name_list = ['Totaloperatingrevenueps_qoq_qoq']
    # for f in factor_name_list:
    #     del_factor_from_panel(panel_path, f)

    # panel_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    # add_fs_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\增加的因子\截面数据'
    #
    # f_list = os.listdir(add_fs_path)
    # for fn in f_list:
    #     f_name = fn.split('.')[0]
    #     print(f_name)
    #     dat = pd.read_csv(os.path.join(add_fs_path, fn), engine='python')
    #     dat = dat.set_index(dat.columns[0])
    #     add_to_panels(dat, panel_path, f_name)



