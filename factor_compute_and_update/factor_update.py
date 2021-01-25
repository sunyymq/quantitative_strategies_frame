import os
import warnings
import calendar
import numpy as np
import pandas as pd
import re
from datetime import datetime, time
from utility.tool0 import Data
from utility.constant import root_dair, data_dair
from utility.relate_to_tushare import generate_months_ends


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class Factor_Update:
    def __init__(self, factor_path, save_path, sentinel=1000, update_only=False):
        self.data = Data()
        self.sentinel = sentinel
        # if not update_only:
        #     self.dates_d = sorted(self.adjfactor.columns)
        #     self.dates_m = sorted(self.pct_chg_M.columns)
        self.save_path = save_path
        self.factor_path = factor_path

    def __getattr__(self, name):
        return getattr(self.data, name, None)

    # 对特定日期的数据添加月度百分比变动
    def fill_pct_monthly(self, target_date=None):
        f_list = os.listdir(self.save_path)
        if not target_date:
            f_d = f_list[-1]
            target_date = datetime.strptime(f_d.split('.')[0], "%Y-%m-%d")
        elif target_date.strftime("%Y-%m-%d") + '.csv' in f_list:
            f_d = target_date.strftime("%Y-%m-%d") + '.csv'
        else:
            print('未找到该日期的文件')
            raise KeyError

        data = pd.read_csv(os.path.join(self.save_path, f_d), engine='python',
                           encoding='gbk')

        data = data.set_index('Code')

        # data[] 'is_open1' 'PCT_CHG_NM'

        PCT_CHG_NM = self.data.PCT_CHG_NM/100
        if target_date not in PCT_CHG_NM.columns:
            raise KeyError
            print('未在PCT_CHG_NM因子中找到{}的数据'.format(target_date))

        is_open = self.data.IS_OPEN1

        data['Pct_chg_nm'] = PCT_CHG_NM[target_date]
        data['Is_open1'] = is_open[target_date]
        data = data.reset_index()
        data = data.set_index('No')
        data.index.name = 'No'
        data.to_csv(os.path.join(self.save_path, f_d), encoding='gbk')

    # 创建一个新月份的因子表格
    def create_factor_file(self, date, path=None):
        # 得到基本数据
        dat0 = self.get_basic_data(date)
        # 得到因子数据
        dat1 = self.get_factor_data(date)
        # 拼接
        dat2 = self.concat_df(dat0, dat1)

        dat2.index.name = 'code'
        datdf = dat2.reset_index()
        datdf.index = range(1, len(datdf) + 1)
        datdf.index.name = 'No'

        res = self.rename_columns(datdf)
        # 保存
        savepath = os.path.join(self.save_path, date.strftime("%Y-%m-%d")+'.csv')
        self.save_file(res, savepath)

    @staticmethod
    def rename_columns(dat_df):
        # 统一所有列名的命名标准，都改成首字母大小，后续字母小写的格式。
        zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查中文
        rename_dict = {}
        for col in dat_df.columns:

            if col == 'name':
                rename_dict.update({'name': 'Sec_name'})
                continue

            match = zhmodel.search(col)
            if not match:   # 不包含中文
                new_c = col[0].upper() + col[1:].lower()
                rename_dict.update({col: new_c})
            else:
                pass

        rename_dict.update({'中信一级行业': 'Industry_zx'})
        rename_dict.update({'申万一级行业': 'Industry_sw'})
        rename_dict.update({'申万二级行业': 'Second_industry'})
        dat_df = dat_df.rename(columns=rename_dict)

        return dat_df

    def save_file(self, datdf, path):

        for col in ['Sec_name', 'Industry_sw']:
            datdf[col] = datdf[col].apply(str)
        # datdf = datdf.loc[~datdf['Sec_name'].str.contains('0')]

        save_cond1 = (~datdf['Sec_name'].str.contains('ST'))  # 剔除ST股票
        save_cond2 = (~pd.isnull(datdf['Mkt_cap_float']))     # 剔除市值为空的股票
        save_cond = save_cond1 & save_cond2
        datdf = datdf.loc[save_cond]

        print('因子数据保存完毕，地址及文件名为{}'.format(path))
        return datdf.to_csv(path, encoding='gbk')

    def concat_df(self, dat_basic, dat_factor):

        res = pd.concat([dat_basic, dat_factor], axis=1)
        return res

    # 得到股票的基本面信息
    def get_basic_data(self, date):
        # code, name, ipo_date, industry_sw, MKT_CAP_FLOAT, is_open1, PCT_CHG_NM
        stock_basic = self.data.stock_basic_inform
        mkt_cap_float = self.data.MKT_CAP_FLOAT * 10000
        pct_chg_nm = self.data.PCT_CHG_NM / 100
        res = stock_basic[['SEC_NAME', 'MKT', 'IPO_DATE', '中信一级行业', '申万一级行业', '申万二级行业']]

        res = self.concat_df(res, pd.DataFrame({'MKT_CAP_FLOAT': mkt_cap_float[date]}))
        res = self.concat_df(res, pd.DataFrame({'PCT_CHG_NM': pct_chg_nm[date]}))

        return res

    # 得到因子数据
    def get_factor_data(self, date):

        code_list = self.data.stock_basic_inform.index
        datdf = pd.DataFrame(index=code_list)

        basic_name = ['MKT_CAP_FLOAT', 'PCT_CHG_NM']

        factor_names = os.listdir(self.factor_path)
        fns = [f.split('.')[0] for f in factor_names if f.split('.')[0] not in basic_name]

        not_in_fn = ['pct_chg_nw'.upper()]

        fns_end = [f for f in fns if f not in not_in_fn]

        for fn in fns_end:
            res = eval('self.data.' + fn)
            if date in res.columns:
                datdf[fn] = res.loc[code_list, date]
            else:
                datdf[fn] = None
        return datdf


# 得到最后一个的更新日期
def get_latest_updated_date():
    data = Data()
    c = data.closeprice_daily
    o = data.openprice_daily
    adj = data.adjfactor
    mv = data.negotiablemv_daily
    pe = data.pe_daily

    res = np.min([c.columns[-1], o.columns[-1], adj.columns[-1], mv.columns[-1], pe.columns[-1]])

    return res


def history_factor_rename():
    history_path = [r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子',
                    r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子']

    # history_path = [r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子-副本']

    for path in history_path:
        fn_list = os.listdir(path)
        for fn in fn_list:
            p = os.path.join(path, fn)
            tmp_df = pd.read_csv(p, encoding='gbk', engine='python')
            if 'No' in tmp_df.columns:
                tmp_df = tmp_df.set_index('No')
            new_df = Factor_Update.rename_columns(tmp_df)
            new_df.to_csv(p, encoding='gbk')


# 生成新的月度因子表格
def generate_new_factor_table(latest_date=None):
    save_path = os.path.join(root_dair, '因子预处理模块', '因子')
    factor_path = os.path.join(data_dair, 'factor_data')

    # latest_date = get_latest_updated_date()

    z = Factor_Update(save_path=save_path, factor_path=factor_path)
    if not latest_date:
        mes = generate_months_ends()
        latest_date = mes[-1]

    # 对上一个月的数据添加月度价格变动百分比数据
    z.fill_pct_monthly()
    z.create_factor_file(latest_date)


if __name__ == "__main__":
    generate_new_factor_table()







