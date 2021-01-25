# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:56:01 2018

@author: admin
"""
import os
import warnings
import numpy as np
import pandas as pd
from utility.constant import data_dair
import copy
import pandas.tseries.offsets as toffsets
from dask import dataframe as dd
warnings.filterwarnings('ignore')


# 根目录地址，在constant文件里面定义
ROOT_PATH = data_dair


# 日期格式调整
def ensure_time(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return x


# 常用的辅助工具类，可通过属性的方式读取数据，定义了保存数据函数，地址通过constant里面的data_dair定义
class Data:
    '''
    以属性的方式获得已存的基础数据，用来计算因子数据。
    '''

    global ROOT_PATH
    save_root = ROOT_PATH
    pathmap = {}
    
    def __init__(self):

        # 跟目录
        path_list = [ROOT_PATH,]

        # 通过递归遍历的方式，得到文件名和地址的字典，填充pathmap
        for path in path_list:
            self._all_files_path(path)

        # 数据名称
        self._ori_factors = sorted(self.pathmap.keys())

        # 定义保存地址
        self.save_path = os.path.join(self.save_root, "factor_data")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    # 通过递归遍历的方式，得到文件名和地址的字典，填充self.pathmap
    def _all_files_path(self, rootDir):
        for root, dirs, files in os.walk(rootDir):   # 分别代表根目录、文件夹、文件
            for file in files:                       # 遍历文件
                self.pathmap.update({self.__name_modify(file): (root, file)})
            for dir in dirs:                         # 遍历目录下的子目录
                dir_path = os.path.join(root, dir)   # 获取子目录路径
                self._all_files_path(dir_path)       # 递归调用

    # 去掉后缀
    def __name_modify(self, f_name):
        if f_name.endswith('.csv') or f_name.endswith('.xlsx'):
            f_name = f_name.split('.')[0]
            if all(n.isnumeric() for n in f_name[-10:].split('-')):
                f_name = f_name[:-11]
        return f_name            # f_name.lower()

    # 通过self.pathmap字典，得到保存数据的地址，再读取文件
    def _open_file(self, name, **kwargs):        
        path, file_name = self.pathmap[name]

        ext = file_name.split('.')[-1]
        if ext == 'csv' or ext == 'CSV':
            dat = self.__read_csv(path, file_name, name, **kwargs)
        elif ext == 'xlsx':
            dat = self.__read_excel(path, file_name, name, **kwargs)
        else:
            msg = f"不支持的文件存储类型：{ext}"
            raise TypeError(msg)
        return dat

    # 读取csv
    def __read_csv(self, path, rname, fname, **kwargs):
        read_conds = {
                'encoding': 'gbk',
                # 'encoding': 'UTF-8',
                'engine': 'python',
                'index_col': [0],
                **kwargs
                }
#       if 'applied_lyr_date_d' in rname or 'applied_rpt_date_d' in rname:
#           read_conds['usecols'] = lambda s: (START_TIME < s < END_TIME) or (s == 'Code')
        if fname == 'indexquote_changepct':
            read_conds['encoding'] = 'UTF-8'
        dat = pd.read_csv(os.path.join(path, rname), **read_conds)
        try:
            dat.columns = pd.to_datetime(dat.columns)
        except Exception as e:
            try:
                dat.index = pd.to_datetime(dat.index)
                dat = dat.T
            except Exception as e:
                pass

        if fname in ('close',):
            dat = dat.where(dat != 0, np.nan)
        if fname in ('stm_issuingdate', 'applied_rpt_date_M'):
            dat = dat.where(dat != '0', pd.NaT)
            dat = dat.applymap(ensure_time)
        return dat

    # 读取excel
    def __read_excel(self, path, rname, fname, **kwargs):
        path = os.path.join(path, rname)
        
        if fname == 'all_stocks_code':
            kwargs['parse_dates'] = ['ipo_date', "delist_date"]
        elif fname == 'all_index_code':
            kwargs['parse_dates'] = ['PubDate', 'EndDate']
        else:
            kwargs['index_col'] = [0]
        dat = pd.read_excel(path, encoding='gbk', **kwargs)
        
        if fname not in ('all_stocks_code', 'all_index_code', 
                         'month_map', 'month_group'):
            try:
                dat.columns = pd.to_datetime(dat.columns)
            except Exception as e:
                # print('非日期格式')
                pass

        if fname in ('cogs_q', 'ebitps_q', 'ev2_m', 'net_profit_ttm_q',
                     'netprofit_report_q', ):
            dat = dat.where(dat != 0, np.nan)
        
        if fname in ('month_map', 'month_group'):
            dat = dat.set_index(['calendar_date'])
        return dat

    # 保存数据
    def save(self, df, name, save_path=None, **kwargs):
        if not save_path:
            save_path = self.save_path
        path = os.path.join(save_path, name)
        if name.endswith('csv'):
            df.to_csv(path, encoding='gbk', **kwargs)
        elif name.endswith('xlsx'):
            df.to_excel(path, encoding='gbk', **kwargs)
        elif '.' not in path:
            df.to_csv(path+'.csv', encoding='gbk', **kwargs)
        else:
            print('不能识别的存储类型')

        print(f'Save {name} successfully.')
        print(f'存储地址为 {path}.')

    # 定义__getattr__魔法函数，改变寻找属性的路径
    def __getattr__(self, name, **kwargs):
        # 先查看 self.__dict__字典
        if name not in self.__dict__:
            name = self._get_fac_valid_name(name)
            res = self._open_file(name, **kwargs)
            self.__dict__[name] = res
        return self.__dict__[name]

    # 数据名称查找
    def _get_fac_valid_name(self, name):

        if name not in self._ori_factors:
            i = 0
            while True:
                try:
                    cur_fname = self._ori_factors[i]
                    if cur_fname.startswith(name):
                        name = cur_fname
                        break
                    i += 1
                except IndexError:
                    msg = f"请确认因子名称{name}是否正确"
                    raise Exception(msg)

        return name

    # 根据是wind还是juyuan, 重新设定index的codes形式
    def reindex(self, df, to='wind', if_index=False):
        dat = df.copy()
        if all('.' in code for code in dat.index) and to == 'wind':
            return dat
        if all(code.startswith('`') for code in dat.index) and to == 'juyuan':
            return dat
        
        if if_index:
            all_codes = getattr(self, 'all_index_code',)
        else:
            all_codes = getattr(self, 'all_stocks_code',)
        if to == 'wind':
            idx_code = 'juyuan_code'
        else:
            idx_code = 'wind_code'
        code_map = all_codes[['juyuan_code', 'wind_code']].set_index(idx_code) 
        new_idx = code_map.loc[df.index]
        new_idx_val = new_idx.values.flatten()
        dat.index = np.where(pd.isna(new_idx_val), new_idx.index, new_idx_val)
        return dat

    @staticmethod
    # 同比增长率
    def generate_yoygr(dat):
        res = pd.DataFrame()
        for i in range(4, len(dat.columns)):
            col1 = dat[dat.columns[i]]
            col2 = dat[dat.columns[i-4]]
            res_tmp = (col1-col2)/col2
            res_tmp[col1 < 0] = np.nan
            res_tmp[col2 < 0] = np.nan
            res_tmp.name = dat.columns[i]
            res = pd.concat([res, res_tmp], axis=1)

        return res

        # self.my_added_finance_datapath

    @staticmethod
    # 价格数据转化为收益率数据
    def price_to_ret(dat_df, axis=0):
        res = (dat_df - dat_df.shift(1, axis=axis)) / dat_df.shift(1, axis=axis)
        return res

    # 环比增长率
    def generate_qoqgr(self, dat):
        res = pd.DataFrame()
        for i in range(1, len(dat.columns)):
            col1 = dat[dat.columns[i]]
            col2 = dat[dat.columns[i - 1]]
            res_tmp = (col1 - col2) / col2
            res_tmp[col1 < 0] = np.nan
            res_tmp[col2 < 0] = np.nan
            res_tmp.name = dat.columns[i]
            res = pd.concat([res, res_tmp], axis=1)

        return res

    # 环比差值
    @staticmethod
    def generate_diff(dat):
        res = pd.DataFrame()
        for i in range(1, len(dat.columns)):
            col1 = dat[dat.columns[i]]
            col2 = dat[dat.columns[i - 1]]
            res_tmp = (col1 - col2)
            res_tmp.name = dat.columns[i]
            res = pd.concat([res, res_tmp], axis=1)

        return res


def scaler(se, scaler_max, scaler_min):
    # 归一化的算法逻辑
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min

    tmp = se.sort_values(ascending=False)

    tmp_non = tmp[tmp.index[pd.isna(tmp)]]
    tmp = tmp.dropna()
    # 先自然排序，把收益率序列换成N到1的排序，收益率越高越大，
    after_sort = pd.Series(range(len(tmp), 0, -1), index=tmp.index)

    # 然后再归一化
    res = (after_sort - after_sort.min())/(after_sort.max() - after_sort.min())
    res = res*(scaler_max - scaler_min) + scaler_min

    res = pd.concat([res, tmp_non])

    return res


# 把最新一期的股票池数据写入txt文件，以便wind导入“我的股票池”模块
def add_stock_pool_txt(stocks_list, save_name, pool_to_wind_path=None, renew=False):
    '''
    如果有重复的股票，wind导入时会自动过滤掉。
    在txt文件中，有空行不影响导入，但是没有换行，会影响导入
    '''
    if not pool_to_wind_path:
        pool_to_wind_path = r'D:\pythoncode\IndexEnhancement\股票池'
    if renew:
        if os.path.exists(os.path.join(pool_to_wind_path, save_name + '.txt')):
            os.remove(os.path.join(pool_to_wind_path, save_name + '.txt'))

    if not os.path.exists(pool_to_wind_path):
        os.makedirs(pool_to_wind_path)

    # 打开模式a+：打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。
    # 文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
    f = open(os.path.join(pool_to_wind_path, save_name + '.txt'), "a+")
    pool = f.readlines()
    for s in stocks_list:
        pool.append(s+"\n")
    f.writelines(pool)
    f.close()  # 关闭文件


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


if __name__ == "__main__":
    # 测试代码
    data = Data()
    stock_basic_inform = data.stock_basic_inform





