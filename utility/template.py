import pandas as pd
import shelve
import numpy as np
import calendar
from collections import defaultdict
from utility.tool0 import Data
from datetime import datetime
from utility.analysis import BackTest, bool_2_ones


# 选股策略模版，后续的选股策略可继承该类
class SelectStockTemplate:
    def __init__(self):
        self._data = Data()
        self.freq = 'M'

    @staticmethod
    # 通过带有不同权重的股票池得到每期股票池内股票的名字
    def pool_2_secname(pool):
        data = Data()
        stock_basic = data.stock_basic_inform
        all_stocks_code = stock_basic[['SEC_NAME']]

        # 存储每次更新财务数据时的结果
        maxl = np.sum(pool > 0, axis=0).max()
        res = pd.DataFrame(index=range(0, maxl))

        # 对选出来的股票向后移动一期，因为回测是根据月度收益模式，日期为月末最后一天，而公告日期为月末最后一天，
        # 所有需要向后移动一期。
        for col, items in pool.iteritems():
            selected = items[items.index[items > 0]]
            selected = all_stocks_code.loc[selected.index, 'sec_name'.upper()]
            selected = pd.DataFrame(selected.values, columns=[col])

            res = pd.concat([res, selected], axis=1)

        return res

    # 因子过滤方式，选择一致预期数据月末相对上月调升的标的
    def factor_filter(self, factor_name, neutralized=False, filter_type='raise'):
        if neutralized:
            print('还未实现，需要读中性化处理后的文件夹，再一个一个的记录')
        else:
            factor_df = eval('self._data.' + factor_name)

        res_pool = pd.DataFrame()
        # 过滤模式
        if filter_type == 'raise':
            for i in range(1, len(factor_df.columns)):
                prior_col = factor_df[factor_df.columns[i-1]]
                new_col = factor_df[factor_df.columns[i]]
                tmp_df = pd.DataFrame({factor_df.columns[i]: new_col > prior_col})

                res_pool = pd.concat([res_pool, tmp_df], axis=1)

        return res_pool

    # 两个DF合并，如果不是Bool型，则先转换成Bool型
    def pool_inter(self, pool_1, pool_2):

        # 当df里面是float，而需要转换成bool型的时候的辅助函数，对为0的转换为false,非0的数据转换成True
        if type(pool_1.values[0, 0]) != bool:
            pool_1 = pool_1.applymap(lambda x: True if x != 0.0 else False)
        if type(pool_2.values[0, 0]) != bool:
            pool_2 = pool_2.applymap(lambda x: True if x != 0.0 else False)

        ret = pool_1 & pool_2
        return ret

    # 把values为bool的df,转换为等权配置的df
    def equal_allocation(self, pool, min_num=None):
        ret = pool / pool.sum()
        return ret

    # 对stock_pool中的每列，如果其股票数量大于num,则按照factor_name排序，选择排名前num的股票
    # stock_pool中的value需要为True或者是False，如果是float型，则先转换成bool型
    def factor_top_n(self, stock_pool, factor, num_or_percent, top_or_down='top', indus_m=None):

        if type(stock_pool.values[0, 0]) != bool:
            stock_pool_tmp = stock_pool.applymap(lambda x: True if x != 0.0 else False)
        else:
            stock_pool_tmp = stock_pool

        if isinstance(factor, str):
            fac = eval('self._data.' + factor)
        elif isinstance(factor, pd.DataFrame):
            fac = factor

        baisc = self._data.stock_basic_inform
        ret_pool = pd.DataFrame()
        # fac和stock_pool的列求并集
        new_columns = [col for col in stock_pool_tmp.columns if col in fac.columns]
        stock_pool_tmp = stock_pool_tmp[new_columns]

        if top_or_down in ['top', 'TOP']:
            ascending_rank = False
        elif top_or_down in ['down', 'DOWN']:
            ascending_rank = True

        for col, se in stock_pool_tmp.iteritems():

            if num_or_percent > 1:
                num = num_or_percent
            # 输入为百分比，则选择出给定因子的排名靠前或考后的一定比例的股票
            elif num_or_percent < 1:
                num = int((se > 0).sum() * num_or_percent)

            if se.sum() > num:
                try:
                    fac_tmp = fac[col]
                except Exception as e:
                    print('debugggggg')

                tmp_df = pd.concat([se, fac_tmp], axis=1)
                tmp_df.columns = ['tof', 'fn']
                # 按照fn排序
                tmp_df = tmp_df.sort_values(by='fn', ascending=ascending_rank)
                # 如果对一级行业的个股数量有要求
                if indus_m:
                    # 使用排序后的数据加入一级行业，每个行业最多选择M个公司，多余M的个换成False
                    tmp_df['一级'] = baisc.loc[tmp_df.index, '申万一级行业']
                    grouped = tmp_df.groupby('一级')
                    new_tmp_df = pd.DataFrame()
                    for ind, se in grouped:
                        if se['tof'].sum() > indus_m:
                            se['cumsum'] = se['tof'].cumsum()
                            se.loc[se.index[se['cumsum'] > indus_m], 'tof'] = False
                            new_tmp_df = pd.concat([new_tmp_df, se[['tof', 'fn']]], axis=0)
                        else:
                            new_tmp_df = pd.concat([new_tmp_df, se[['tof', 'fn']]], axis=0)
                else:
                    new_tmp_df = tmp_df

                # 按照fn排序
                new_tmp_df = new_tmp_df.sort_values(by='fn', ascending=ascending_rank)

                # new_tmp_df = tmp_df
                # 累加求和
                new_tmp_df['cumsum'] = new_tmp_df['tof'].cumsum()
                # 数量超出的股票设为False
                new_tmp_df.loc[new_tmp_df.index[new_tmp_df['cumsum'] > num], 'tof'] = False

                ret_pool = pd.concat([ret_pool, pd.DataFrame({col: new_tmp_df['tof']})], axis=1)

            else:
                ret_pool = pd.concat([ret_pool, pd.DataFrame({col: se})], axis=1)

        # ret_pool.sum()
        return ret_pool

    # 剔除部分不符合选股逻辑的个股，添加符合选股逻辑的个股
    def codes_replace(self, codes_list, path=None):

        if not path:
            path = r'D:\Database_Stock\股票池_最终\个股替换.xlsx'
        codes_repalce_df = pd.read_excel(path, encoding='bgk')

        del_list = list(codes_repalce_df['剔除'])
        add_list = list(codes_repalce_df['添加'])

        res = [i for i in codes_list if i not in del_list]

        while np.nan in add_list:
            add_list.remove(np.nan)

        for i in add_list:
            if i not in res:
                res.append(i)

        return res

    # 最新股票池持仓的个股基本面信息，包括行业、市值、盈利、成长性、一致预期、估值
    def pool_infor(self, codes_list):
        basic_inform = self._data.stock_basic_inform
        negotiable = self._data.negotiablemv_daily
        pe = self._data.pe_daily
        roe_ttm = self._data.roettm
        basicepsyoy = self._data.basicepsyoy
        west_netprofit_yoy = self._data.west_netprofit_yoy

        res = pd.DataFrame()
        res = pd.concat([res, basic_inform.loc[codes_list, ['SEC_NAME', '申万一级行业']]], axis=1)

        t = pd.DataFrame({'流通市值': negotiable.loc[codes_list, negotiable.columns[-2]]/10000})
        res = pd.concat([res, t], axis=1)

        t = pd.DataFrame({'PE': pe.loc[codes_list, pe.columns[-2]]})
        res = pd.concat([res, t], axis=1)

        t = pd.DataFrame({'roe_ttm': roe_ttm.loc[codes_list, roe_ttm.columns[-1]]})
        res = pd.concat([res, t], axis=1)

        t = pd.DataFrame({'basicepsyoy': basicepsyoy.loc[codes_list, basicepsyoy.columns[-1]]})
        res = pd.concat([res, t], axis=1)

        t = pd.DataFrame({'west_netprofit_yoy': west_netprofit_yoy.loc[codes_list, west_netprofit_yoy.columns[-1]]})
        res = pd.concat([res, t], axis=1)

        return res

    # 使用未来数据，剔除交易日涨跌停或停牌的股票
    def eliminate_un_open(self, pool):
        is_open = self._data.IS_OPEN1

        open_tmp = pd.DataFrame()
        # 得到下一个交易周期的开始日期和结束日期
        for i in range(0, len(pool.columns)-1, 1):
            st = np.where(is_open.columns == pool.columns[i])[0][0]
            et = np.where(is_open.columns == pool.columns[i+1])[0][0]

            tmp_df = is_open.iloc[:, st+1:et+1]
            tmp_df.all(axis=1)
            tmp_se = tmp_df.iloc[:, 0] & tmp_df.iloc[:, -1]

            open_tmp = pd.concat([open_tmp, pd.DataFrame({pool.columns[i]: tmp_se})], axis=1)

        # 针对最后一期
        loc = np.where(is_open.columns == pool.columns[-1])[0][0]
        tmp_se = is_open.iloc[:, loc+1]
        open_tmp = pd.concat([open_tmp, pd.DataFrame({pool.columns[-1]: tmp_se})], axis=1)

        res = open_tmp & pool
        return res

    # 回测函数，使用回测类进行
    def backtest(self, stock_pool, freq='M', bench='WindA', plt=True, fee='No_fee'):

        bt = BackTest(stock_pool, freq, adjust_freq=freq, fee_type=fee,  # 'fee',  'No_fee'
                      benchmark_str=bench)
        bt.run_bt()
        ana, each_year, each_month = bt.analysis()

        nv = bt.net_value
        fig = bt.plt_pic(show=plt)

        pool_secname = self.pool_2_secname(stock_pool)

        res_dict = {'指标': ana,
                    '净值': nv,
                    '历年表现': each_year,
                    '每月表现': each_month,
                    '净值走势图': fig,
                    '每期股票名称': pool_secname,
        }

        return res_dict


if '__main__' == __name__:
    template = SelectStockTemplate()
    template.factor_filter('WEST_NETPROFIT_YOY')



