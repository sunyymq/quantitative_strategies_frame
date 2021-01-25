# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 08:26:53 2019

@author: admin
"""

import os
import calendar
import warnings
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.tseries.offsets as toffsets
from itertools import dropwhile, chain, product
from functools import reduce, wraps
from dask import dataframe as dd
from numba import jit
from dask.multiprocessing import get
# from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from utility.tool0 import Data
from utility.tool3 import adjust_months, add_to_panels, append_df
from utility.relate_to_tushare import generate_months_ends

warnings.filterwarnings('ignore')

START_YEAR = 2009
END_YEAR = 2019
BENCHMARK = '000300.SH'
VERSION = 6

dat = Data()
work_dir = os.path.dirname(__file__)
SENTINEL = 1e10

__spec__ = None


def get_signal_season_value(dat_df):
    sig_season_va = pd.DataFrame(None, index=dat_df.index, columns=dat_df.columns)

    for i in range(0, len(dat_df.columns)):
        col = dat_df.columns[i]
        if col.month == 3:  # 一季度数据维持不变
            sig_season_va[col] = dat_df[col]
        else:
            if i > 0:
                sig_season_va[col] = dat_df[col] - dat_df[dat_df.columns[i - 1]]

    for i in range(0, len(sig_season_va.columns)):
        col = sig_season_va.columns[i]
        if sig_season_va[col].isnull().sum() / len(sig_season_va[col]) > 0.8:
            continue
        else:
            break

    if i > 0:
        sig_season_va.drop(sig_season_va.columns[0:i], axis=1, inplace=True)

    return sig_season_va


def get_season_mean_value(dat_df):

    res = pd.DataFrame(None, index=dat_df.index, columns=dat_df.columns)

    for i in range(0, len(dat_df.columns)):
        col = dat_df.columns[i]
        # 一季度数据维持不变
        if col.month == 3 or i == 0:
            res[col] = dat_df[col]
        # 后面三个季度都取前两个季度的均值
        else:
            if (dat_df.columns[i] - dat_df.columns[i-1]).days > 4*30:
                res[col] = dat_df[col]
            else:
                res[col] = dat_df[dat_df.columns[i - 1:i+1]].mean(axis=1)

    return res


# 计时器装饰器
def time_decorator(func):
    @wraps(func)
    def timer(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        end = datetime.datetime.now()
        print(f'“{func.__name__}” run time: {end - start}.')
        return result
    return timer


# 延迟初始化属性，第一次调用时不在__dict__里，调用相关函数，第二次调用时已经在__dict__里了，不再重复计算
class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            # 把instance中的 self.func.__name__ 属性值设定为 value
            setattr(instance, self.func.__name__, value)
            return value


class parallelcal:
    '''
    常用函数
    '''

    @staticmethod
    # 常用的回归函数
    def _regress(y, X, intercept=True, weight=1, verbose=True):
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if intercept:
            cols = X.columns.tolist()
            X['const'] = 1
            X = X[['const'] + cols] 
        try:
            model = sm.WLS(y, X, weights=weight)
            result = model.fit()
            params = result.params
        except Exception as e:
            print('debug')

        # 是否返回残存
        if verbose:
            resid = y - pd.DataFrame(np.dot(X, params), index=y.index, 
                         columns=y.columns)
            if intercept:
                return params.iloc[1:], params.iloc[0], resid
            else:
                return params, None, resid
        else:
            if intercept:
                return params.iloc[1:]
            else:
                return params
    
    @staticmethod
    def weighted_std(series, weights):
        return np.sqrt(np.sum((series-np.mean(series)) ** 2 * weights))
    
    def weighted_func(self, func, series, weights):
        weights /= np.sum(weights)
        if func.__name__ == 'std':
            return self.weighted_std(series, weights)
        else:
            return func(series * weights)
    
    def nanfunc(self, series, func, sentinel=SENTINEL, weights=None):
        valid_idx = np.argwhere(series != sentinel)
        if weights is not None:
            return self.weighted_func(func, series[valid_idx], 
                                      weights=weights[valid_idx])
        else:
            return func(series[valid_idx])
    
    @staticmethod
    def _cal_cmra(series, months=12, days_per_month=21, version=6):
        z = sorted(series[-i * days_per_month:].sum() for i in range(1, months+1))
        if version == 6:
            return z[-1] - z[0]
        elif version == 5:
            return np.log(1 + z[-1]) - np.log(1 + z[0])

    # barra定义的中位数市值
    def _cal_midcap(self, series):
        x = series.dropna().values
        y = x ** 3
        beta, alpha, _ = self._regress(y, x, intercept=True, weight=1, verbose=True)
        resid = series ** 3 - (alpha + beta[0] * series)
        return resid



    # 计算流动性函数
    @staticmethod
    def _cal_liquidity(series, days_pm=21, sentinel=-SENTINEL):
        freq = len(series) // days_pm
        valid_idx = np.argwhere(series != SENTINEL)
        series = series[valid_idx]
        res = np.log(np.nansum(series) / freq)
        if np.isinf(res):
            return sentinel
        else:
            return res
    
    def _cal_growth_rate(self, series): 
        valid_idx = np.argwhere(pd.notna(series))
        y = series[valid_idx]
        x = np.arange(1, len(series)+1)[valid_idx]
        
        coef = self._regress(y, x, intercept=True, verbose=False)
        return coef.iloc[0] / y.mean()
    
    def _get_apply_rptdate(self, df, idate=None, delist_map=None):        
        code = df.name
        delist_date = delist_map[code]
        rptrealdates = idate.loc[code,:].tolist()
        
        if pd.isnull(delist_date): 
            res = [self.__append_date(rptrealdates, curdate, idate) for curdate in df.index]
        else:
            res = []
            for curdate in df.index:
                if curdate >= delist_date:
                    res.append(pd.NaT)
                else:
                    res.append(self.__append_date(rptrealdates, curdate, idate))
        return res
    
    @staticmethod
    def __append_date(rptrealdates, curdate, idate, base_time='1899-12-30 00:00:00'):
        base_time = pd.to_datetime(base_time)
        rptavaildates = sorted(d for d in rptrealdates if d < curdate and d != base_time)
        if rptavaildates:
            availdate1 = rptavaildates[-1]
            didx = rptrealdates.index(availdate1) 
            try:
                availdate2 = rptavaildates[-2]
            except IndexError:
                pass
            else:
               if availdate1 == availdate2:
                   didx += 1
            finally:
               return idate.columns[didx]
        else:
            return pd.NaT 


def get_fill_vals(nanidx, valid_vals):
    start, end = nanidx[0], nanidx[-1]
    before_val, after_val = valid_vals[start-1], valid_vals[end+1]
    diff = (after_val - before_val) / (1 + len(nanidx))
    fill_vals = [before_val + k * diff for k in range(1, len(nanidx) + 1)]
    return fill_vals


def linear_interpolate(series):
    vals = series.values
    valid_vals = list(dropwhile(lambda x: np.isnan(x), vals))
    idx = np.where(np.isnan(valid_vals))[0]
    start_idx = len(vals) - len(valid_vals)
    
    tmp = []
    for i, cur_num in enumerate(idx):
        try:
            next_num = idx[i+1]
        except IndexError:
            if cur_num < len(vals) - 1:
                try:
                    if tmp:
                        tmp.append(cur_num)
                        fill_vals = get_fill_vals(tmp, valid_vals)
                        for j in range(len(tmp)):
                            vals[start_idx + tmp[j]] = fill_vals[j]
                    else:
                        fill_val = 0.5 * (valid_vals[cur_num - 1] + valid_vals[cur_num + 1])
                        vals[start_idx + cur_num] = fill_val
                except IndexError:
                    break
                break
        else:
            if next_num - cur_num == 1:
                tmp.append(cur_num)
            else:
                if tmp:
                    tmp.append(cur_num)
                    fill_vals = get_fill_vals(tmp, valid_vals)
                    for j in range(len(tmp)):
                        vals[start_idx + tmp[j]] = fill_vals[j]
                    tmp = []
                else:
                    try:
                        fill_val = 0.5 * (valid_vals[cur_num - 1] + valid_vals[cur_num + 1])
                        vals[start_idx + cur_num] = fill_val
                    except IndexError:
                        break
    res = pd.Series(vals, index=series.index)
    return res


# TODO: hfq_close


# # 计算su的特定方法，用于计算sue,revsue
# def _calculate_su(x_df):
#     '''
#     计算逻辑为当前值减去预期值，预期值计算方法为去年同期值加上一个漂移量。
#     漂移量计算方法为过去8期中每期值与上期值得差值的均值。
#     最后结果在处以过去7期得标准差。
#     若财务数据是剔除了一季度的数据，那么下面的硬编码就是对的，如果是未做过处理的，下面的硬编码就是错的。
#     '''
#
#     chazhi = x_df - x_df.shift(1, axis=1)
#     chazhi = chazhi.iloc[:, 0:-1]
#     exp = np.nanmean(chazhi, axis=1)
#     s = np.nanstd(chazhi, axis=1)
#     # 使用 x_df.iloc[:, -2] 加预期值是使用上个季度的数据加上一个单季度收入增长的均值，
#     res = (x_df.iloc[:, -1] - (x_df.iloc[:, -2] + exp))/s
#
#     return res


# 计算su的特定方法，用于计算sue,revsue
def _calculate_su_simple(x_df):
    '''
    直接用当季度减去去年同期数据，在比上过去4期的方差
    '''

    res = (x_df.iloc[:, -1] - x_df.iloc[:, -5])/np.nanstd(x_df.iloc[:, -5:], axis=1)
    # if np.any(x_df.iloc[:, -5] < 0):
    #     print('here')
    res[x_df.iloc[:, -5] < 0] = 0
    return res


# 逻辑关系： CALFUNC继承了Data, 各个大类因子又继承了CALFUNC，大类因子下面的小类因子，是大类因子的函数【属性】。


class CALFUNC(Data):    
    def __init__(self):
        super().__init__()
        self._parallel_funcs = parallelcal()

    # 先找实例的__dict__里有没有adjfactor，没有再在CALFUNC的__dict__里，没有再在父类Data里找，Data里定义了__getattr__方法，
    # 运行相应的函数，然后再运行lazyproperty里面的setattr。
    @lazyproperty
    def tdays(self):
        return sorted(self.adjfactor.columns)

    def __getattr__(self, item):
        try:
            return getattr(self._parallel_funcs, item)
        except AttributeError:
            return super().__getattr__(item)

    # 日频率的数据转化为月频率的数据
    @staticmethod
    def d_freq_to_m_freq(dat, shift=False):
        # 如果 shift为True，表示使用下一个交易日的数据来做月末数据
        me = generate_months_ends()
        new_cols = [m for m in me if m in dat.columns]

        if shift:
            dat = dat.shift(-1, axis=1)

        new_dat = dat[new_cols]

        return new_dat

    # 剔除开始年份以前的数据
    @staticmethod
    def del_dat_early_than(dat, year):
        new_cols = [d for d in dat.columns if d.year >= year]
        new_dat = dat[new_cols]
        return new_dat

    def _cal_pctchange_in_month(self, series):
        date = series.name
        stocks = series.index
        if date.month > 1:
            lstyear = date.year
            lstmonth = date.month - 1
        else:
            lstyear = date.year - 1
            lstmonth = date.month - 1 + 12
        lstday = min(date.day, calendar.monthrange(lstyear, lstmonth)[1])
        lstdate = toffsets.datetime(lstyear, lstmonth, lstday)
        lstdateidx = self._get_date_idx(lstdate, self.tdays)
        lstdate = self.tdays[lstdateidx]
        try:
            res = self.hfq_close.loc[stocks, date] / self.hfq_close.loc[stocks, lstdate] - 1
        except KeyError:
            res = series.where(pd.isnull(series), np.nan)
        return res 
    
    def _cal_pctchange_nextmonth(self, series):
        date = series.name
        stocks = series.index
        td_idx = self._get_date_idx(date, self.tdays)
        nstart_idx, nend_idx = td_idx + 1, td_idx + 21
        try:
            nend_date = self.tdays[nend_idx]
        except IndexError:
            return np.array([np.nan] * len(series))
        else:
            nstart_date = self.tdays[nstart_idx]
            res = self.hfq_close.loc[stocks, nend_date] / self.hfq_close.loc[stocks, nstart_date] - 1
            return res
    
    def _get_price_last_month_end(self, type_='close'):
        price = getattr(self, type_,)
        if price is None:
            raise Exception(f'Unsupported price type {type_}!')
        date_range = price.columns.tolist()
        price_me = price.T.groupby(pd.Grouper(freq='m')).apply(lambda df:df.iloc[-1])
        dates_me = [d2 for d1, d2 in zip(date_range[1:], date_range[:-1]) if d1.month != d2.month]
        if len(dates_me) < price_me.shape[0]:
            price_me.index = dates_me + date_range[-1:]
        else:
            price_me.index = dates_me
        price_lme = price_me.reindex(date_range).fillna(method='ffill').shift(1)
        return price_lme
    
    def _get_pct_chg_m_daily(self):
        tdays = dropwhile(lambda date: date.year != START_YEAR - 6, self.tdays)
        res = pd.DataFrame(index=self.hfq_close.index, 
                           columns=list(tdays))
#        return self._pandas_parallelcal(res, self._cal_pctchange_in_month, 
#               args=(self._get_date_idx, self.tdays, self.hfq_close), axis=0).T
        return res.apply(self._cal_pctchange_in_month).T
    
    def _get_pct_chg_nm(self):
        tdays = dropwhile(lambda date: date.year != START_YEAR - 5, self.tdays)
        res = pd.DataFrame(index=self.hfq_close.index, 
                           columns=list(tdays))
        return res.apply(self._cal_pctchange_nextmonth).T
    
    @staticmethod
    def clear_vals(df):
        df.iloc[1:] = np.nan
        return df
            
    @staticmethod
    def fill_vals(df):
        return df.fillna(method='ffill')

    # 数据清理
    def clean_data(self, datdf, index=False, limit_days=False):
        # pandas freely uses object dtype when columns contain mixed values (strings, numbers, nan), means object
        if datdf.index.dtype != 'O':
            datdf = datdf.T
        # index设为wind代码
        data_cleaned = self.reindex(datdf, to='wind', if_index=index)
        if not index:
            valid_stks = [i for i in data_cleaned.index if i[0].isnumeric()]
            data_cleaned = data_cleaned.loc[valid_stks, :]
        if limit_days:
            tdays = self.get_trade_days(START_YEAR, END_YEAR)
            data_cleaned = data_cleaned.loc[:, tdays]
        # 在return的时候转置了
        return data_cleaned.T

    def _get_intact_rpt_dates(self, start_year=START_YEAR, end_year=END_YEAR):
        intact_rpt_dates = sorted(map(lambda x: pd.to_datetime(f'{x[0]}-{x[1]}'), 
            product(range(start_year, end_year+1), ('03-31', '06-30', '09-30', '12-31'))))
        cur_year = toffsets.datetime.now().year
        cur_month = toffsets.datetime.now().month
        if end_year == cur_year:
            if cur_month < 4:
                return intact_rpt_dates[:-4]
            elif 4 <= cur_month < 8:
                return intact_rpt_dates[:-3]
            elif 8 <= cur_month < 10:
                return intact_rpt_dates[:-2]
            else:
                return intact_rpt_dates[:-1]
        return intact_rpt_dates
        
    def _get_ttm_data(self, datdf):
        datdf = self.clean_data(datdf)
        rpt_dates = sorted(d for d in datdf.index if (d.month, d.day) in ((3, 31), (6, 30), (9, 30), (12, 31)))
        datdf = datdf.loc[rpt_dates,:]
        
        start_year, end_year = rpt_dates[0].year, rpt_dates[-1].year
        intact_rpt_dates = self._get_intact_rpt_dates(start_year, end_year)
        datdf = datdf.reindex(intact_rpt_dates)
        virtual_rpt_dates = np.argwhere(pd.isnull(datdf).sum(axis=1) == datdf.shape[1])
        datdf.iloc[virtual_rpt_dates.flatten()] = 0
        
        res = pd.DataFrame(columns=datdf.index, index=datdf.columns)
        for date in datdf.index[4:]:
            if date.month == 12:
                res[date] = datdf.loc[date]
                continue
            lst_rpt_y = pd.to_datetime(f'{date.year - 1}-12-31')
            lst_rpt_q = pd.to_datetime(f'{date.year - 1}-{date.month}-{date.day}')
            res[date] = datdf.loc[lst_rpt_y] + datdf.loc[date] - datdf.loc[lst_rpt_q]
        return res.T

    def get_trade_days(self, start_year=START_YEAR, end_year=END_YEAR, tdays=None):
        if tdays is None:
            tdays = self.tdays    
        start_idx = self._get_date_idx(f'{start_year}-01-01', tdays)
        if end_year < toffsets.date.today().year:
            end_idx = self._get_date_idx(f'{end_year}-12-31', tdays)
        else:
            end_idx = -1
        return tdays[start_idx+1:end_idx]
    
    def _shift(self, datdf, shift=1):
        datdf = self.clean_data(datdf)
        datdf = datdf.shift(shift)
        return datdf

    def mv_weighted_ret(self, negotiablemv, change_pct):
        return change_pct*negotiablemv/negotiablemv.sum()

    @staticmethod
    def get_smb_vmg(self, negotiablemv, ep, change_pct):
        # 计算逻辑：
        # 构建FF3回归的几个x
        # 首先排除市值最小的30%，剩余的80%的股票为评价因子模型的股票池。将股票之中的股票按照市值大小分成Small和Big两组、按照
        # EP分成Value、Middle以及Growth
        # 形成  S/V, S/M, S/G
        #       B/V, B/M, B/G
        # SMB = (S/V + S/M + S/G)/3 - (B/V + B/M + B/G)/3
        # VMG = (S/V + B/V)/2 - (S/G + B/G)/2
        # MKT 使用沪深300

        # 输入: 市值、估值,
        # 输出: SMB, VMG

        smb = pd.Series(index=negotiablemv.columns)
        vmg = pd.Series(index=negotiablemv.columns)
        smb_sample = pd.Series(index=negotiablemv.columns)
        vmg_sample = pd.Series(index=negotiablemv.columns)

        for col in negotiablemv.columns:
            print(col)
            section_nego = negotiablemv[col]
            section_ep = ep[col]
            section_change_pct = change_pct[col]
            section_nego.dropna(inplace=True)
            section_ep.where(section_ep > 0, 0, inplace=True)
            # 剔除最小的30%，
            section_nego = section_nego.sort_values(ascending=False)  # 由大至小排序
            # todel = int(len(section_nego)*0.10)
            # section_nego.drop(section_nego.index[-todel:], inplace=True)
            # section_ep = section_ep[section_nego.index]
            # 分组
            big = set(section_nego.index[:int(len(section_nego)*0.1)])    # 市值大的10%
            small = set(section_nego.index[int(len(section_nego)*0.1):])  # 市值小的10%
            section_ep = section_ep.sort_values(ascending=False)
            value = set(section_ep.index[:int(len(section_ep)*0.1)])      # ep大的前1/10,即估值低的
            growth = set(section_ep.index[int(len(section_ep)*0.1):])

            # sr = self.mv_weighted_ret(section_nego[small], section_change_pct[small])
            # br = self.mv_weighted_ret(section_nego[big], section_change_pct[big])
            # vr = self.mv_weighted_ret(section_nego[value], section_change_pct[value])
            # gr = self.mv_weighted_ret(section_nego[value], section_change_pct[growth])

            smb_sample[col] = section_change_pct[small].mean() - section_change_pct[big].mean()
            vmg_sample[col] = section_change_pct[value].mean() - section_change_pct[growth].mean()

            # smb_weighted[col] = sr - br
            # vmg_weighted[col] = sr - br

        return smb_sample, vmg_sample


    @staticmethod
    def __drop_invalid_and_fill_val(series, val=None, method=None):
        valid_idx = np.argwhere(series.notna()).flatten()
        try:
            series_valid = series.iloc[valid_idx[0]:]
        except IndexError:
            return series
        if val:
            series_valid = series_valid.fillna(val)
        elif method:
            series_valid = series_valid.fillna(method=method)
        else:
            median = np.nanmedian(series_valid)
            series_valid = series_valid.fillna(median)
        series = series.iloc[:valid_idx[0]].append(series_valid)
        return series
    
    def _fillna(self, datdf, value=None, method=None):
        datdf = self.clean_data(datdf)        
        datdf = datdf.apply(self.__drop_invalid_and_fill_val, 
                            args=(value, method))
        return datdf
    
    @staticmethod
    def _get_exp_weight(window, half_life):
        exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_wt[::-1] / np.sum(exp_wt)
    
    @staticmethod
    @time_decorator
    def _pandas_parallelcal(dat, myfunc, ncores=6, args=None, axis=1, window=None):
        '''
        用rolling加myfunc的形式解决问题，myfunc必须自己定义
        '''
        if axis == 0 and window is None:
            dat = dat.T
        dat = dd.from_pandas(dat, npartitions=ncores)
        if window:
            dat = dat.rolling(window=window)
            if args is None:
                res = dat.apply(myfunc)
            else:
                res = dat.apply(myfunc, args=args)
        else:
            res = dat.apply(myfunc, args=args, axis=1)
        return res.compute(scheduler='threads')
    
    @time_decorator
    def _get_growth_rate(self, ori_data, periods=5, freq='y'):
#        self = CALFUNC(); s = parallelcal(); freq='y'; periods=5;ori_data = self.totalassets
        ori_data = self.clean_data(ori_data)
        current_lyr_rptdates = self.applied_lyr_date_d
        if ori_data.index.dtype == 'O':
            ori_data = ori_data.T
        ori_data = ori_data.groupby(pd.Grouper(freq=freq)).apply(lambda df: df.iloc[-1])
        ori_data = self._pandas_parallelcal(ori_data, self._cal_growth_rate, window=5)
        
        current_lyr_rptdates = current_lyr_rptdates.loc[ori_data.columns, :] 
        current_lyr_rptdates = current_lyr_rptdates.stack().reset_index()
        current_lyr_rptdates.columns = ['code', 'date', 'rptdate']
        current_lyr_rptdates['rptdate'] = pd.to_datetime(current_lyr_rptdates['rptdate'])
        current_lyr_rptdates = current_lyr_rptdates.set_index(['code', 'rptdate'])
        
        ori_data = ori_data.T.stack()
        res = ori_data.loc[current_lyr_rptdates.index]
        res = pd.concat([current_lyr_rptdates, res], axis=1)
        res = res.reset_index()
        res.columns = ['code', 'rptdate', 'date', 'value']
        res = pd.pivot_table(res, values='value', index=['code'], columns=['date'])
        return res

    # stocks中在date日前上市的股票范围
    def _get_codes_listed(self, stocks, date):
        stk_basic_info = self.stock_basic_inform
        stk_basic_info = stk_basic_info[stk_basic_info.index.isin(stocks)]
        stk_basic_info['ipo_date'.upper()] = pd.to_datetime(stk_basic_info['ipo_date'.upper()])
        listed_cond = stk_basic_info['ipo_date'.upper()] <= date
        res = stk_basic_info[listed_cond].index.tolist()
        return res

    # stocks中在date日前未退市的股票
    def _get_codes_not_delisted(self, stocks, date):
        stk_basic_info = self.stock_basic_inform
        stk_basic_info = stk_basic_info[stk_basic_info.index.isin(stocks)]
        stk_basic_info['ipo_date'.upper()] = pd.to_datetime(stk_basic_info['ipo_date'.upper()])
        stk_basic_info['delist_date'.upper()] = pd.to_datetime(stk_basic_info['delist_date'.upper()])

        # 硬编码了，[0]是平安银行
        not_delisted_cond = ((stk_basic_info['delist_date'.upper()] == stk_basic_info['delist_date'.upper()][0]) |
                             (stk_basic_info['delist_date'.upper()] >= date))
        res = stk_basic_info[not_delisted_cond].index.tolist()
        return res


    # 通过设定code为benchmark的 _get_ret 来提取指数价格变动数据
    def _get_benchmark_ret(self, code=BENCHMARK):        
        # pct_chg_idx = self.clean_data(self.indexquote_changepct / 100,
        #                               index=True)
        # idx_ret = self._get_ret(pct_chg_idx, [code])

        data = Data()
        index_daily = data.index_daily
        index_price = index_daily.loc[code, :]
        ret = index_price/index_price.shift(1) - 1
        ret.dropna(inplace=True)
        ret = pd.DataFrame(ret.values, index=ret.index, columns=[code])

        return ret

    # 选择 pct_chg 中 codes 的价格变动，codes为NaN的话无用
    def _get_ret(self, pct_chg=None, codes=None):
        if pct_chg is None:
            pct_chg = self.clean_data(self.changepct / 100)
        if codes is None:
            codes = pct_chg.columns
        ret = pct_chg.loc[:, codes]
        return ret
    
    def _rolling(self, datdf, window, half_life=None, 
                 func_name='sum', weights=None):
        global SENTINEL
        datdf = self.clean_data(datdf)
        datdf = datdf.where(pd.notnull(datdf), SENTINEL)
        if datdf.index.dtype == 'O':
            datdf = datdf.T
        
        func = getattr(np, func_name, )
        if func is None:
            msg = f"""Search func:{func_name} from numpy failed, 
                   only numpy ufunc is supported currently, please retry."""
            raise AttributeError(msg)
        
        if half_life or (weights is not None):
            exp_wt = self._get_exp_weight(window, half_life) if half_life else weights
            args = func, SENTINEL, exp_wt
        else:
            args = func, SENTINEL
        
        try:
            res = self._pandas_parallelcal(datdf, self.nanfunc, args=args, 
                                           axis=0, window=window)
        except Exception:
            print('Calculating under single core mode...')
            res = self._rolling_apply(datdf, self.nanfunc, args=args, 
                                      axis=0, window=window)
        return res.T

    def _rolling_apply(self, datdf, func, args=None, axis=0, window=None):
        if window:
            res = datdf.rolling(window=window, axis=axis).apply(func, args=args)
        else:
            res = datdf.apply(func, args=args, axis=axis)
        return res

    def _rolling_regress(self, y, x, window=5, half_life=None,
                         intercept=True, verbose=False, fill_na=0, target_date=None):
        fill_args = {'method': fill_na} if isinstance(fill_na, str) else {'value': fill_na}

        stocks = y.columns
        if half_life:
            weight = self._get_exp_weight(window, half_life)
        else:
            weight = 1
        # np.flatten() #默认按行的方向降维, 返回一个一维数组

        if not target_date:                  # 若未规定目标日期，则是得到一个日频的循环，计算量很大
            rolling_ys = rolling_windows(y, window)
            rolling_xs = rolling_windows(x, window)
        else:
            rolling_ys = []
            rolling_xs = []
            for t_d in target_date:
                loc = np.where(y.index == t_d)[0][0]
                rolling_ys.append(y.iloc[loc + 1 - window:loc+1, :].values)
                rolling_xs.append(x.iloc[loc + 1 - window:loc+1].values)

        bet = pd.DataFrame()
        alpha = pd.DataFrame()
        sigma = pd.DataFrame()
        # enumerate 形成带 i 的一个迭代器
        for i, (rolling_x, rolling_y) in enumerate(zip(rolling_xs, rolling_ys)):
            if not target_date:                  # 若未规定目标日期，则是得到一个日频的循环，计算量很大
                tmp_index = y.index[i:i+window]
            else:
                loc = np.where(y.index == target_date[i])[0][0]
                tmp_index = y.index[loc + 1 - window:loc+1]

            rolling_y = pd.DataFrame(rolling_y, columns=y.columns,
                                     index=tmp_index)
            # 开头结尾两个日期
            window_sdate, window_edate = rolling_y.index[0], rolling_y.index[-1]
            # 开始时已经上市的股票和结尾时未退市的股票的交集
            stks_to_regress = sorted(set(self._get_codes_listed(stocks, window_sdate)) & \
                              set(self._get_codes_not_delisted(stocks, window_edate)))
            # nan处理
            rolling_y = rolling_y[stks_to_regress].fillna(**fill_args)
            # rolling_y.shape
            try:
                b, a, resid = self._regress(rolling_y.values, rolling_x,
                                        intercept=True, weight=weight, verbose=True)
            except:
                print(i)
                raise
            vol = np.std(resid, axis=0)
            vol.index = a.index = b.columns = stks_to_regress
            b.index = [window_edate]
            vol.name = a.name = window_edate
            bet = pd.concat([bet, b], axis=0)
            alpha = pd.concat([alpha, a], axis=1)
            sigma = pd.concat([sigma, vol], axis=1)
        
        bet = bet.T
        return bet, alpha, sigma

    # CAPM模型，用股票收益对基准指数收益做回归，得到beta、alpha和sigma, sigma为残差的方差
    def _capm_regress(self, window=504, half_life=252):
        y = self._get_ret(self.changepct / 100)
        x = self._get_benchmark_ret()
        beta, alpha, sigma = self._rolling_regress(y, x, window=window, 
                                                   half_life=half_life)
        return beta, alpha, sigma

    def _get_period_d(self, date, offset=None, freq=None, datelist=None):
        if isinstance(offset, (float, int)) and offset > 0:
            raise Exception("Must return a period before current date.")
        
        conds = {}
        freq = freq.upper()
        if freq == "M":
            conds.update(months=-offset)
        elif freq == "Q":
            conds.update(months=-3*offset)
        elif freq == "Y":
            conds.update(years=-offset)
        else:
            freq = freq.lower()
            conds.update(freq=-offset)
        
        sdate = pd.to_datetime(date) - pd.DateOffset(**conds) 
        
        if datelist is None:
            datelist = self.tdays
        sindex = self._get_date_idx(sdate, datelist, ensurein=True)
        eindex = self._get_date_idx(date, datelist, ensurein=True)
        return datelist[sindex:eindex+1]
    
    def _get_date_idx(self, date, datelist=None, ensurein=False):
        msg = """Date {} not in current tradedays list. If tradedays list has already been setted, \
              please reset tradedays list with longer periods or higher frequency."""
        date = pd.to_datetime(date)
        if datelist is None:
            datelist = self.tdays
        try:
            datelist = sorted(datelist)
            idx = datelist.index(date)
        except ValueError:
            if ensurein:
                raise IndexError(msg.format(str(date)[:10]))
            dlist = list(datelist)
            dlist.append(date)
            dlist.sort()
            idx = dlist.index(date) 
            if idx == len(dlist)-1 or idx == 0:
                raise IndexError(msg.format(str(date)[:10]))
            return idx - 1
        return idx




# 1、Size
class Size(CALFUNC):
    @lazyproperty
    def LNCAP(self):  
        lncap = np.log(self.negotiablemv * 10000)
        return lncap
    # 5 --
    @lazyproperty
    def MIDCAP(self):
        lncap = self.LNCAP
        midcap = self._pandas_parallelcal(lncap, self._cal_midcap, axis=0).T
        return midcap
    # 5 --


# 2、Volatility
class Volatility(CALFUNC):    
    @lazyproperty
    def BETA(self, version=VERSION):
        if 'BETA' in self.__dict__:
            return self.__dict__['BETA']
        if version == 6:
            beta, alpha, hsigma = self._capm_regress(window=504, half_life=252)
            self.__dict__['HSIGMA'] = hsigma
            self.__dict__['HALPHA'] = alpha
        elif version == 5:
            beta, alpha, hsigma = self._capm_regress(window=252, half_life=63)
            self.__dict__['HSIGMA'] = hsigma
        return beta
    #5 ** window = 252, hl = 63

    @lazyproperty
    def HSIGMA(self, version=VERSION):
        if 'HSIGMA' in self.__dict__:
            return self.__dict__['HSIGMA']
        if version == 6:
            beta, alpha, hsigma = self._capm_regress(window=504, half_life=252)
            self.__dict__['BETA'] = hsigma
            self.__dict__['HALPHA'] = alpha
        elif version == 5:
            beta, alpha, hsigma = self._capm_regress(window=252, half_life=63)
            self.__dict__['BETA'] = beta
        return hsigma

    #5 ** window = 252, hl = 63
    @lazyproperty
    def HALPHA(self):
        if 'HALPHA' in self.__dict__:
            return self.__dict__['HALPHA']
        beta, alpha, hsigma = self._capm_regress(window=504, half_life=252)
        self.__dict__['BETA'] = beta
        self.__dict__['HSIGMA'] = hsigma
        return alpha
    
    @lazyproperty
    def DASTD(self):
        dastd = self._rolling(self.changepct / 100, window=252, 
                              half_life=42, func_name='std')
        return dastd

    #5 --
    @lazyproperty
    def CMRA(self, version=VERSION):
        stock_ret = self._fillna(self.changepct / 100, 0)
        if version == 6:
            ret = np.log(1 + stock_ret)
        elif version == 5:
            index_ret = self._get_benchmark_ret()
            index_ret = np.log(1 + index_ret)
            index_ret, stock_ret = self._align(index_ret, stock_ret)
            ret = np.log(1 + stock_ret).sub(index_ret[BENCHMARK], axis=0)
        cmra = self._pandas_parallelcal(ret, self._cal_cmra, args=(12, 21, version), 
                                        window=252, axis=0).T
        return cmra
    #5 ** cmra = ln(1+zmax) - ln(1+zmin), z = sigma[ln(1+rt) - ln(1+r_hs300)]


# TODO:turnovervalue合成的问题

# 3、Liquidity
class Liquidity(CALFUNC):
    @lazyproperty
    def STOM(self):
        amt, mkt_cap_float = self._align(self.turnovervalue, self.negotiablemv)
        share_turnover = amt / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stom = self._pandas_parallelcal(share_turnover, self._cal_liquidity, 
                                        axis=0, window=21).T
        return stom
    #5 --     
    @lazyproperty
    def STOQ(self):
        amt, mkt_cap_float = self._align(self.turnovervalue, self.negotiablemv)
        share_turnover = amt / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stoq = self._pandas_parallelcal(share_turnover, self._cal_liquidity, 
                                        axis=0, window=63).T
        return stoq
    #5 --
    @lazyproperty
    def STOA(self):
        self = CALFUNC()
        amt, mkt_cap_float = self._align(self.turnovervalue, self.negotiablemv)
        share_turnover = amt / mkt_cap_float
        stoa2 = share_turnover.rolling(12*21).apply(self._cal_liquidity)
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stoa = self._pandas_parallelcal(share_turnover, self._cal_liquidity, 
                                        axis=0, window=252).T
        stoa.T.loc['2009-01-05':'2009-01-23',:]
        stoa2.loc['2009-01-05':'2009-01-23',:]
        return stoa
    #5 --
    @lazyproperty
    def ATVR(self):
        turnoverrate = self.turnoverrate / 100
        atvr = self._rolling(turnoverrate, window=252, half_life=63, func_name='sum')
        return atvr


# 4、Momentum
class Momentum(CALFUNC):
    @lazyproperty
    def STREV(self):
        strev = self._rolling(self.changepct / 100, window=21, 
                              half_life=5, func_name='sum')
        return strev
        
    @lazyproperty
    def SEASON(self):
        nyears = 5
        pct_chg_m_d = self._get_pct_chg_m_daily()
        pct_chgs_shift = [pct_chg_m_d.shift(i*21*12 - 21) for i in range(1,nyears+1)]
        seasonality = sum(pct_chgs_shift) / nyears
        seasonality = seasonality.loc[f'{START_YEAR}':f'{END_YEAR}'].T
        return seasonality
        
    @lazyproperty
    def INDMOM(self):
        window = 6 * 21; half_life = 21
        logret = np.log(1 + self._fillna(self.changepct / 100, 0))
        rs = self._rolling(logret, window, half_life, 'sum')
        
        cap_sqrt = np.sqrt(self.negotiablemv)
        ind_citic_lv1 = self.firstind
        rs, cap_sqrt, ind_citic_lv1 = self._align(rs, cap_sqrt, ind_citic_lv1)
        
        dat = pd.DataFrame()
        for df in [rs, cap_sqrt, ind_citic_lv1]:
            df.index.name = 'time'
            df.columns.name = 'code'
            dat = pd.concat([dat, df.unstack()], axis=1)

        dat.columns = ['rs', 'weight', 'ind']
        dat = dat.reset_index()
        
        rs_ind = {(time, ind): (df['weight'] * df['rs']).sum() / df['weight'].sum()
                  for time, df_gp in dat.groupby(['time']) 
                  for ind, df in df_gp.groupby(['ind'])}
        
        def _get(key):
            nonlocal rs_ind
            try:
                return rs_ind[key]
            except:
                return np.nan
            
        dat['rs_ind'] = [_get((date, ind)) for date, ind in zip(dat['time'], dat['ind'])]
        dat['indmom'] = dat['rs_ind'] - dat['rs'] * dat['weight'] / dat['weight'].sum()
        indmom = pd.pivot_table(dat, values='indmom', index=['code'], columns=['time'])
        return indmom

    @lazyproperty
    def RSTR(self, version=VERSION):
        benchmark_ret = self._get_benchmark_ret()
        stock_ret = self.changepct / 100
        benchmark_ret, stock_ret = self._align(benchmark_ret, stock_ret)
        benchmark_ret = benchmark_ret[BENCHMARK]
        
        excess_ret = np.log((1 + stock_ret).divide((1 + benchmark_ret), axis=0))
        if version == 6:
            rstr = self._rolling(excess_ret, window=252, half_life=126, func_name='sum')
            rstr = rstr.rolling(window=11, min_periods=1).mean()
        elif version == 5:
            exp_wt = self._get_exp_weight(504+21, 126)[:504]
            rstr = self._rolling(excess_ret.shift(21), window=504, weights=exp_wt, 
                                 func_name='sum')
        return rstr
    #5 ** window=504, l=21, hl=126

    # 盈余动量
    def SUE(self):
        # 使用原始的财务数据
        eps = self.basiceps
        # 得到单季度的数据。
        sig_season_va = get_signal_season_value(eps)
        cols = pd.DataFrame([i for i in sig_season_va.columns])

        sue = pd.DataFrame()
        rolling_cols = rolling_windows(cols, 6)
        for roll in rolling_cols:
            res = _calculate_su_simple(sig_season_va[roll])
            res = pd.DataFrame(res.values, index=res.index, columns=[roll[-1]])
            sue = pd.concat([sue, res], axis=1)

        sue.dropna(how='all', axis=0, inplace=True)

        sue = adjust_months(sue)
        sue = append_df(sue)

        return sue

    # 营收动量
    def REVSU(self):
        netprofit = self.totaloperatingrevenueps
        # 得到单季度的数据。
        sig_season_va = get_signal_season_value(netprofit)
        cols = pd.DataFrame([i for i in sig_season_va.columns])

        revsu = pd.DataFrame()
        rolling_cols = rolling_windows(cols, 6)
        for roll in rolling_cols:
            res = _calculate_su_simple(sig_season_va[roll])
            res = pd.DataFrame(res.values, index=res.index, columns=[roll[-1]])
            revsu = pd.concat([revsu, res], axis=1)

        revsu.dropna(how='all', axis=0, inplace=True)

        revsu = adjust_months(revsu)
        revsu = append_df(revsu)

        return revsu

    # FF3因子回归中的临时变量
    def FF3_Daily(self):
        y = self._get_ret(self.changepct_daily / 100)
        x_pb = self.ep_daily
        x_mv = self.negotiablemv_daily

        x_pb, x_mv, y = self._align(x_pb, x_mv, y)
        x_pb = x_pb.T
        x_mv = x_mv.T
        y = y.T

        smb_daily, vmg_daily= self.get_smb_vmg(self, x_mv, x_pb, y)

        return smb_daily, vmg_daily


    # 残差动量：一月、三月、六月
    def RESMOM(self):
        '''
        过去1年数据进行 FF 三因子回归[基准走势，流通市值，PB]，最近1月残差均值/标准差
        这个数据可以是日频的，也可以是月频的。我只使用到月频数据，所以用月频的。
        '''
        month_ends = generate_months_ends()
        y = self._get_ret(self.changepct_daily / 100)
        x_bench = self._get_benchmark_ret()
        smb_daily = self.SMB_DAILY
        vmg_daily = self.VMG_DAILY
        smb_daily = smb_daily.T
        vmg_daily = vmg_daily.T
        y = y.T
        mut_dates = list(vmg_daily.index & smb_daily.index & x_bench.index & y.index)
        smb_daily = smb_daily.loc[mut_dates, :]
        vmg_daily = vmg_daily.loc[mut_dates, :]
        x_bench = x_bench.loc[mut_dates, :]
        y = y.loc[mut_dates, :]

        resid_pd_1m, resid_pd_3m, resid_pd_6m = self._rolling_regress_for_ff3(y, x_bench, smb_daily, vmg_daily,
                                                                              dates=month_ends, window=225)
        return resid_pd_1m, resid_pd_3m, resid_pd_6m


# 5、Quality
class Quality(CALFUNC):
    pass


# TODO：preferedequity 没有
class Leverage(Quality):
    @lazyproperty
    def MLEV(self, version=VERSION):
#            longdebttoequity, be = self._align(self.longdebttoequity, self.totalshareholderequity)
#            ld = be * longdebttoequity
        if version == 6:
            method = 'lyr' 
        elif version == 5:
            method = 'mrq'
        ld = self._transfer_freq(self.totalnoncurrentliability, 
                                 method=method, from_='q', to_='d')
        pe = self._transfer_freq(self.preferedequity, 
                                 method=method, from_='q', to_='d')
        me = self._shift(self.totalmv, shift=1)
        me, pe, ld = self._align(me, pe, ld)
        mlev = (me + pe + ld) / me
        return mlev.T
    #5 ** pe, ld ---- mrq
    @lazyproperty
    def BLEV(self, version=VERSION):
        if version == 6:
            method = 'lyr' 
        elif version == 5:
            method = 'mrq'
        ld = self._transfer_freq(self.totalnoncurrentliability, 
                                 method=method, from_='q', to_='d')
        pe = self._transfer_freq(self.preferedequity, 
                                 method=method, from_='q', to_='d')
        be = self._transfer_freq(self.totalshareholderequity, 
                                 method=method, from_='q', to_='d')
        be, pe, ld = self._align(be, pe, ld)
        blev = (be + pe + ld) / be
        return blev.T
    #5 ** oe, ld, be ---- mrq
    @lazyproperty
    def DTOA(self, version=VERSION):
        if version == 6:
            tl = self._transfer_freq(self.totalliability, 
                                     method='lyr', from_='q', to_='d')
            ta = self._transfer_freq(self.totalassets,
                                     method='lyr', from_='q', to_='d')
            tl, ta = self._align(tl, ta)
            dtoa = tl / ta
        elif version == 5:
            sewmi_to_ibd, sewmit_to_tl, tl = self._align(self.sewmitointerestbeardebt, 
                                           self.sewithoutmitotl, self.totalliability)
            ibd = tl * (sewmit_to_tl / sewmi_to_ibd)
            ta, td = self._align(self.totalassets, ibd)
            dtoa = td / ta
            dtoa = self._transfer_freq(dtoa, method='mrq', from_='q', to_='d')
        return dtoa.T
    #5 ** dtoa = td / ta; td -- long-term debt+current liabilities;td,ta ---- mrq


class EarningsVariablity(Quality):
    window = 5
    @lazyproperty
    def VSAL(self):
        sales_y = self._transfer_freq(self.operatingreenue, None, 
                                      from_='q', to_='y')
        std = sales_y.rolling(window=self.window).std() 
        avg = sales_y.rolling(window=self.window).mean()
        vsal = std / avg
        
        vsal = self._transfer_freq(vsal, method='lyr', from_='q', to_='d')
        return vsal.T
    
    @lazyproperty
    def VERN(self):
        earnings_y = self._transfer_freq(self.netprofit, None, 
                                         from_='q', to_='y')
        std = earnings_y.rolling(window=self.window).std() 
        avg = earnings_y.rolling(window=self.window).mean()
        vern = std / avg
        
        vern = self._transfer_freq(vern, method='lyr', from_='q', to_='d')
        return vern

    @lazyproperty
    def VFLO(self):
        cashflows_y = self._transfer_freq(self.cashequialentincrease, None, 
                                          from_='q', to_='y')
        std = cashflows_y.rolling(window=self.window).std()
        avg = cashflows_y.rolling(window=self.window).mean()
        vflo = std / avg
        
        vflo = self._transfer_freq(vflo, method='lyr', from_='q', to_='d')
        return vflo.T
    
#    @lazyproperty
#    def ETOPF_STD(self):
#        etopf = self.west_eps_ftm.T
#        etopf_std = etopf.rolling(window=240).std()
#        close = self.clean_data(self.close)
#        etopf_std, close = self._align(etopf_std, close)
#        etopf_std /= close
#        return etopf_std.T


class EarningsQuality(Quality):
    @lazyproperty
    def ABS(self):
        cetoda, ce = self._align(self.capitalexpendituretodm, self.capital_expenditure) #wind:资本支出/折旧加摊销，资本支出
        cetoda = cetoda.apply(linear_interpolate)
        da = ce / cetoda #此处需对cetoda插值填充处理
        #lc_mainindexdata:归属母公司股东的权益/带息债务(%), 归属母公司股东的权益/负债合计(%), 负债合计
        sewmi_to_ibd, sewmit_to_tl, tl = self._align(self.sewmitointerestbeardebt, 
                                           self.sewithoutmitotl, self.totalliability)
        ibd = tl * (sewmit_to_tl / sewmi_to_ibd)
        
        ta, cash, tl, td = self._align(self.totalassets, self.cashequialents, 
                                       self.totalliability, ibd)
        noa = (ta - cash) - (tl - td)
        
        noa, da = self._align(noa, da)
        accr_bs = noa - noa.shift(1) - da
        
        accr_bs, ta = self._align(accr_bs, ta)
        abs_ = - accr_bs / ta
        abs_ = self._transfer_freq(abs_, method='mrq', from_='q', to_='d')
        return abs_.T
        
    @lazyproperty
    def ACF(self):
        cetoda, ce = self._align(self.capitalexpendituretodm, self.capital_expenditure) #wind:资本支出/折旧加摊销，资本支出
        cetoda = cetoda.apply(linear_interpolate)
        da = ce / cetoda #此处需对cetoda插值填充处理
        ni, cfo, cfi, da = self._align(self.netprofit, self.netoperatecashflow, 
                                       self.netinvestcashflow, da)
        accr_cf = ni - (cfo + cfi) + da
        
        accr_cf, ta = self._align(accr_cf, self.totalassets)
        acf = - accr_cf / ta
        acf = self._transfer_freq(acf, method='mrq', from_='q', to_='d')
        return acf.T
            
class Profitability(Quality):
    @lazyproperty
    def ATO(self):
        sales = self._transfer_freq(self._get_ttm_data(self.operatingreenue),
                                    method='mrq', from_='q', to_='d')
        ta = self._transfer_freq(self.totalassets, method='mrq', 
                                 from_='q', to_='d')
        sales, ta = self._align(sales, ta)
        ato = sales / ta
        return ato.T
    
    @lazyproperty
    def GP(self):
        sales = self._transfer_freq(self.operatingreenue,
                                    method='lyr', from_='q', to_='d')
        cogs = self._transfer_freq(self.cogs_q,
                                   method='lyr', from_='q', to_='d')
        ta = self._transfer_freq(self.totalassets,
                                 method='lyr', from_='q', to_='d')
        sales, cogs, ta = self._align(sales, cogs, ta)
        gp = (sales - cogs) / ta
        return gp.T
        
    @lazyproperty
    def GPM(self):
        sales = self._transfer_freq(self.operatingreenue,
                                    method='lyr', from_='q', to_='d')
        cogs = self._transfer_freq(self.cogs_q,
                                   method='lyr', from_='q', to_='d')
        sales, cogs = self._align(sales, cogs)
        gpm = (sales - cogs) / sales
        return gpm.T
        
    @lazyproperty
    def ROA(self):
        earnings = self._transfer_freq(self._get_ttm_data(self.netprofit),
                                       method='mrq', from_='q', to_='d')
        ta = self._transfer_freq(self.totalassets,
                                 method='mrq', from_='q', to_='d')
        earnings, ta = self._align(earnings, ta)
        roa = earnings / ta
        return roa.T


class InvestmentQuality(Quality):
    window = 5
    @lazyproperty
    def AGRO(self):
        agro = self._get_growth_rate(self.totalassets, periods=self.window, 
                                     freq='y')
        return agro
    
    @lazyproperty
    def IGRO(self):
        igro = self._get_growth_rate(self.totalshares, periods=self.window, 
                                     freq='y')
        return igro
    
    @lazyproperty
    def CXGRO(self):
        cxgro = self._get_growth_rate(self.capital_expenditure, 
                                      periods=self.window, freq='y')
        return cxgro


#6*******Value
class Value(CALFUNC):
    @lazyproperty
    def BTOP(self):
        bv = self._transfer_freq(self.sewithoutmi, method='mrq', from_='q', to_='d')
        bv, mkv = self._align(bv, self.totalmv)
        btop = bv / (mkv * 10000)
        return btop.T
    #5 --
    #*****Earnings Yield
class EarningsYield(Value):
    @lazyproperty
    def ETOP(self):
        earings_ttm = self._transfer_freq(self._get_ttm_data(self.netprofit), 
                                          method='mrq', from_='q', to_='d')
        e_ttm, mkv = self._align(earings_ttm, self.totalmv)
        etop = e_ttm / (mkv * 10000)
        return etop.T
    #5 --
#        @lazyproperty
#        def ETOPF(self):
#            pass
    
    @lazyproperty
    def CETOP(self):
        cash_earnings = self._transfer_freq(self._get_ttm_data(self.netoperatecashflow),
                                            method='mrq', from_='q', to_='d')
        ce, mkv = self._align(cash_earnings, self.totalmv)
        cetop = ce / (mkv * 10000)
        return cetop.T
    #5 --
    @lazyproperty
    def EM(self):
        ebit = self._transfer_freq(self.ebit, method='lyr', from_='q', to_='d')
        
        sewmi_to_ibd, sewmit_to_tl, tl = self._align(self.sewmitointerestbeardebt, 
                                           self.sewithoutmitotl, self.totalliability)
        ibd = tl * (sewmit_to_tl / sewmi_to_ibd)
        ibd = self._transfer_freq(ibd, method='mrq', from_='q', to_='d')
        
        cash = self._transfer_freq(self.cashequialents, method='mrq', from_='q', to_='d')
        ebit, mkv, ibd, cash = self._align(ebit, self.totalmv, ibd, cash) 
        
        ev = mkv * 10000 + ibd - cash
        em = ebit / ev
        return em.T


class LongTermReversal(Value):
    @lazyproperty
    def LTRSTR(self):
        self = CALFUNC()
        benchmark_ret = self._get_benchmark_ret(BENCHMARK)
        stock_ret = self.changepct / 100
        benchmark_ret, stock_ret = self._align(benchmark_ret, stock_ret)
        benchmark_ret = benchmark_ret[BENCHMARK]
        
        excess_ret = np.log((1 + stock_ret).divide((1 + benchmark_ret), axis=0))
        ltrstr = self._rolling(excess_ret, window=1040, half_life=260, func_name='sum').T
        ltrstr = (-1) * ltrstr.shift(273).rolling(window=11).mean()
        return ltrstr.T
    
    @lazyproperty
    def LTHALPHA(self):
        _, alpha, _ = self._capm_regress(window=1040, half_life=260)
        lthalpha = (-1) * alpha.T.shift(273).rolling(window=11).mean()
        return lthalpha.T


#7*******Growth
class Growth(CALFUNC):
    window = 5
#    @lazyproperty
#    def EGRLF(self):
#        pass
    
    @lazyproperty
    def EGRO(self):        
        egro = self._get_growth_rate(self.eps, periods=self.window, 
                                     freq='y')
        return egro
    #5 --
    @lazyproperty
    def SGRO(self):
        total_shares, operatingrevenue = self._align(self.totalshares, self.operatingreenue)
        ops = operatingrevenue / total_shares
        sgro = self._get_growth_rate(ops, periods=self.window,
                                     freq='y')
        return sgro
    #5 -- 
#8*******Sentiment
#class Sentiment(CALFUNC):
#    @lazyproperty
#    def RRIBS(self):
#        pass
#    
#    @lazyproperty
#    def EPIBSC(self):
#        pass
#    
#    @lazyproperty
#    def EARNC(self):
#        pass


#9*******DividendYield
class DividendYield(CALFUNC):
    @lazyproperty
    def DTOP(self):
        dps = self._transfer_freq(self._get_ttm_data(self._fillna(self.dividendps, value=0)),
                                  method='mrq', from_='q', to_='d')
        price_lme = self._get_price_last_month_end('close')
        dps, price_lme = self._align(dps, price_lme)
        dtop = dps / price_lme
        return dtop.T
    
#    @lazyproperty
#    def DPIBS(self):
#        pass


if __name__ == '__main__':

    # 计算一个因子，并添加到现有的因子数据中
    panel_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'

    mom = Momentum()
    # smb_daily, vmg_daily = mom.FF3_Daily()
    res = mom.SUE()
    res.index.name = 'code'
    add_to_panels(res, panel_path, 'SUE')
    mom.save(res, 'SUE')

    # res = mom.REVSU()
    # res.index.name = 'code'
    # add_to_panels(res, panel_path, 'REVSU')
    # mom.save(res, 'REVSU')

    # mom.save(smb_daily, 'SMB_DAILY')
    # mom.save(vmg_daily, 'VMG_DAILY')

    # resid_pd_1m, resid_pd_3m, resid_pd_6m = mom.RESMOM()
    # resid_pd_1m.index.name = 'code'
    # resid_pd_3m.index.name = 'code'
    # resid_pd_6m.index.name = 'code'
    # mom.save(resid_pd_1m, 'RESID_1M')
    # mom.save(resid_pd_3m, 'RESID_3M')
    # mom.save(resid_pd_6m, 'RESID_6M')
    # add_to_panels(resid_pd_1m, panel_path, 'RESID_1M')
    # add_to_panels(resid_pd_3m, panel_path, 'RESID_3M')
    # add_to_panels(resid_pd_6m, panel_path, 'RESID_6M')

    # mom.save(res, 'SUE')
    # res = mom.REVSU()
    # res.index.name = 'code'
    # add_to_panels(res, panel_path, 'REVSU')
    # mom.save(res, 'REVSU')


    '''

    # 大类风格因子，用前面定义的类名来获得，globals().keys()包含前面所有的关键字，从中挑选出首字母大写第二个字母小写的定义的类名
    factor_styles = [name for name in globals().keys()
                     if name[0].isupper() and name[1].islower()
                     and name not in ('In', 'Out', 'Data')]
    cne5 = ['BETA', 'HSIGMA', 'CMRA', 'RSTR', 'MLEV', 'BLEV', 'DTOA']
    for style in factor_styles:

        fstyle = globals()[style]()
        # dir(fstyle) 查看fstyle包含的属性、方法。通过看首字符是否为大写，来得到因子名称。
        factors_names = [name for name in dir(fstyle) if name.isupper()]
        for factor in factors_names:

            if VERSION == 5:
                if factor not in cne5:
                    continue
                save_name = factor+'_5.csv'
            else:
                save_name = factor+'.csv'
            if os.path.exists(os.path.join(fstyle.save_path, save_name)):
                print('{} already exists.'.format(factor+'.csv'))
                continue
            if VERSION == 5:
                print(f'Calculating {factor}_5...')
            else:
                print(f'Calculating {factor}...')
            # getattr是获取对象的特定属性值
            res = fstyle.clean_data(getattr(fstyle, factor), limit_days=True)
            res = res.T
            res = res.applymap(lambda s: float(s))
            res.index.name = 'code'
            cond = (res >= 0) | (~np.isinf(res))
            res = res.where(cond, -SENTINEL)
            fstyle.save(res, save_name)
            print('*'*80)
        '''

#     
# =============================================================================
# #s = CALFUNC()
# #res = s._get_pct_chg_nm()
# #res.T.to_csv('pct_change_nextmonth.csv')        
# =============================================================================


# preferedequity  hfq_close  turnovervalue



