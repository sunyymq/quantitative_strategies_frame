import pandas as pd
import shelve
import numpy as np
import calendar
from collections import defaultdict
from utility.tool0 import Data
from datetime import datetime
from utility.tool2 import ma
from utility.constant import data_dair, root_dair
import os
from utility.constant import code_name_map_citic, code_name_map_sw, index_code_name_map
from utility.relate_to_tushare import stocks_basis, trade_days, generate_months_ends

from utility.select_industry import my_factor_concat
from utility.single_factor_test import get_datdf_in_panel
from utility.template import SelectStockTemplate


class ProfitNoticeSelectStock(SelectStockTemplate):
    def __init__(self, update_only=True, notice_data_path=None):
        super().__init__()

        # 存储各种因子值的地址
        self.stock_factor_path = os.path.join(root_dair, '因子预处理模块', '因子')

        # 存储Wind下载的收益预告数据的地址
        if not notice_data_path:
            self.notice_data_path = r'D:\Database_Stock\Data\profitnotice'
        else:
            self.notice_data_path = notice_data_path

        # 把Wind下载的数据整理成真实的披露月末的形式
        self.announce_dict = None

        self.pool_df = None
        self.save_path = None

    # 处理业绩预告数据，处理逻辑：首先对公告日期修正，修正为当月末的日期
    def profitnotice_process(self, save_path=None):

        mes = generate_months_ends()

        data_path = self.notice_data_path
        # 整理成真实月末日期的dict形式，披露时间月末日为key，相应的df为value
        announce_dict = defaultdict(pd.DataFrame)
        file_list = os.listdir(data_path)
        for f in file_list:
            dat = pd.read_csv(os.path.join(data_path, f), encoding='gbk')
            dat.set_index(dat.columns[0], inplace=True)
            stmp_tmp = f.split('.')[0]
            stmp_dt = datetime.strptime(stmp_tmp, "%Y-%m-%d")

            dat['PROFITNOTICE_FIRSTDATE'] = pd.to_datetime(dat['PROFITNOTICE_FIRSTDATE'])

            if stmp_dt.month == 12:
                print('here')

            # 剔除部分错误的数据
            for i in dat.index:
                dat.loc[i, 'month'] = dat.loc[i, 'PROFITNOTICE_FIRSTDATE'].month

            if stmp_dt.month == 12:
                dat = dat[(dat['month'] == 12) | (dat['month'] == 1)]
            else:
                dat = dat[dat['month'] >= stmp_dt.month]

            dat = dat.drop('month', axis=1)

            # 保留 业绩预增、略增、续盈 的个股
            select = []
            for i in dat.index:
                if dat.loc[i, 'PROFITNOTICE_STYLE'] in ['预增']:  # '续盈', '略增', , '续盈'
                    select.append(i)
            dat = dat.loc[select, :]

            def tmp_fun(mm, mes):
                for m in mes:
                    if mm.year == m.year and mm.month == m.month:
                        return m

            # 逐行遍历，调整为正确的日期
            for code, se in dat.iterrows():
                ana_date = pd.to_datetime(se['PROFITNOTICE_FIRSTDATE'])
                adjust_date = tmp_fun(ana_date, mes)

                if pd.isna(adjust_date):
                    continue

                df = announce_dict[adjust_date]
                df = pd.concat([df, pd.DataFrame({code: se}).T], axis=0)
                announce_dict[adjust_date] = df

        # 整理成key为交易日月末日期，vale为df的dict，看一下数量
        for key, value in announce_dict.items():
            print(key)
            print(len(value))

        self.announce_dict = announce_dict

    # 把dict中value转换成股票池的形式
    def from_dict_2_pool(self):

        pool_df = pd.DataFrame()
        me_list = [key for key in self.announce_dict.keys()]
        print(me_list)
        me_list.sort()
        for me in me_list:
            dat_df = self.announce_dict[me]

            tmp_df = pd.DataFrame(index=dat_df.index, data=True, columns=[me])
            # 判断是否重复，因上市公司可能一个月内出两篇业绩预告，后一篇为修正
            if tmp_df.index.duplicated().any():
                tmp_df = tmp_df.loc[~tmp_df.index.duplicated()]

            pool_df = pd.concat([pool_df, tmp_df], axis=1)

        pool_df.fillna(False, inplace=True)
        pool_df.sum()
        self.pool_df = pool_df

        return pool_df

    # 该策略属于事件型策略，某些月份可能没有事件就导致该其无持仓，但是从交易逻辑上，
    # 前期买入的个股，如果有盈利还是该继续持有的，所以要填充一下
    def pool_fillup(self, pool_tmp, min_num=50):

        res_df = pd.DataFrame()
        pool_tmp.sum()
        pool_tmp.sort_index(axis=1, inplace=True)
        res_df = pd.concat([res_df, pool_tmp[[pool_tmp.columns[0]]]], axis=1)

        for c in range(1, len(pool_tmp.columns)):
            col_peri = pool_tmp.columns[c-1]
            col = pool_tmp.columns[c]
            if pool_tmp[col].sum() < min_num:
                tmp = res_df[col_peri] | pool_tmp[col]

                pool_tmp[col_peri].sum()
                pool_tmp[col].sum()
                tmp.sum()

                res_df = pd.concat([res_df, pd.DataFrame({col: tmp})], axis=1)
            else:
                res_df = pd.concat([res_df, pool_tmp[[col]]], axis=1)

        return res_df

    def run_test(self, test_type='no_filter'):
        self.profitnotice_process()
        pool_notice = self.from_dict_2_pool()
        pool_notice.sum()
        pool_west = self.factor_filter('west', filter_type='raise')
        pool_to_fill = self.pool_inter(pool_notice, pool_west)
        pool_to_fill.sum()

        cols = [c for c in pool_to_fill.columns if c.month in [1, 4, 7, 10]]
        pool_0 = pool_to_fill[cols]
        pool_0.sum().mean()

        # 剔除交易日涨跌停或停牌的股票
        pool = self.eliminate_un_open(pool_0)
        pool.sum().mean()

        # 等权合成初步测试
        if test_type == 'no_filter':
            wei_0 = self.equal_allocation(pool)
            wei_0.sum()
            res_dict = self.backtest(wei_0, freq='Odd', bench='WindA', plt=True, fee='fee')
            res_dict

        elif test_type == 'concat':
            # 合成一个基本面因子来过滤股票
            path_dict = {'save_path': r'D:\Database_Stock\临时\prifitnotice',
                         'factor_panel_path': r'D:\Database_Stock\因子预处理模块\因子（已预处理）',
                        }
            factors_dict = {'Profit_filter': ['Profit_g_q', 'SUE']}
            my_factor_concat(path_dict, factors_dict, status='renew', concat_type='equal_weight',
                             start_date=None)
            path = r'D:\Database_Stock\临时\prifitnotice\新合成因子\因子截面'
            f_datdf = get_datdf_in_panel(path, ['Profit_filter'])

            # 格式转换
            basic_df = pd.DataFrame()
            for k, v in f_datdf.items():
                v.columns = [k]
                basic_df = pd.concat([basic_df, v], axis=1)

            pool_filter_by_basic = self.factor_top_n(pool, factor=basic_df, num_or_percent=50, indus_m=None)
            wei_2 = self.equal_allocation(pool_filter_by_basic)
            (wei_2 > 0).sum()
            pool_secname = SelectStockTemplate.pool_2_secname(wei_2)
            pool_secname.to_csv(r'D:\Database_Stock\临时\每期股票名称_经过基本面过滤.csv', encoding='gbk')
            ana, nv, each_year, fig = self.backtest(wei_2, freq='Odd', bench='WindA', plt=True, fee='No_fee')

        elif test_type == 'sue':
            pool_filter_by_basic = self.factor_top_n(pool, factor='SUE', num_or_percent=50, indus_m=None)
            wei_2 = self.equal_allocation(pool_filter_by_basic)
            (wei_2 > 0).sum()
            wei_2.to_csv(r'D:\Database_Stock\临时\权重.csv')
            pool_secname = SelectStockTemplate.pool_2_secname(wei_2)
            pool_secname.to_csv(r'D:\Database_Stock\临时\每期股票名称_经过基本面过滤.csv', encoding='gbk')
            ana, nv, each_year, fig = self.backtest(wei_2, freq='Odd', bench='WindA', plt=True, fee='No_fee')

        elif test_type == 'sue_rps':
            pool_filter_by_basic = self.factor_top_n(pool, factor='SUE', num_or_percent=50, indus_m=None)
            pool_filter_2 = self.factor_top_n(pool_filter_by_basic, factor='RPS', num_or_percent=30, indus_m=None)
            wei_3 = self.equal_allocation(pool_filter_2)
            pool_secname = SelectStockTemplate.pool_2_secname(wei_3)
            pool_secname.to_csv(r'D:\Database_Stock\临时\每期股票名称SUE过滤再选择RPS大的个股.csv', encoding='gbk')
            ana, nv, each_year, fig = self.backtest(wei_3, freq='Odd', bench='WindA', plt=True, fee='No_fee')

        return ana, nv, each_year, fig

    # 最新的股票池
    def latest_pool(self):
        self.profitnotice_process()
        pool_notice = self.from_dict_2_pool()
        pool_west = self.factor_filter('west', filter_type='raise')
        pool = self.pool_inter(pool_notice, pool_west)

        # 取最后一期
        pool = pool[[pool.columns[-1]]]
        if pool.sum() == 0:
            print('该期没有选出新的个股，可能因当期不在业绩预告期')
            return None
        else:
            pool_filter_1 = self.factor_top_n(pool, factor_name='SUE', num=50, indus_m=None)
            latest_pool = self.factor_top_n(pool_filter_1, factor_name='RPS', num=20, indus_m=None)
            codes_list = list(latest_pool.index[latest_pool[latest_pool.columns[-1]] == True])
            pool_inform = self.pool_infor(codes_list)

            return pool_inform


if __name__ == '__main__':
    pn = ProfitNoticeSelectStock()
    pn.run_test(test_type='sue')




