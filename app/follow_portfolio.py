import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import copy
import sys
import statsmodels.api as sm
from sklearn import preprocessing
from datetime import datetime
import shutil
from functools import reduce
from app.data_management_class import Data_Management
from sklearn.covariance import LedoitWolf
from utility.factor_data_preprocess import add_to_panels, align
from utility.tool0 import Data, add_stock_pool_txt
from utility.tool3 import adjust_months, append_df, wr_excel
from utility.relate_to_tushare import stocks_basis, generate_months_ends, trade_days


# 跟踪报告里面的投资组合表现，计算各种指标
class FollowPortfolio:

    def __init__(self):
        # 更新日度数据
        # data_manage = Data_Management()
        # data_manage.update_market_quote_daily()
        # 辅助类
        self._data = Data()
        # 根目录
        self.basic_path = r'D:\定期报告\策略跟踪报告\组合跟踪'
        # 记录最新持仓和权重的文件
        self.pool_path = os.path.join(self.basic_path, '股票池及权重.xlsx')
        # 记录股票历史收益的文件
        self.his_stock_ret_path = os.path.join(self.basic_path, '股票净值.xlsx')
        # 记录期货历史收益的文件
        self.his_future_ret_path = os.path.join(self.basic_path, '期货净值.xlsx')

        # 文件夹自查
        self.checking()

    def checking(self):
        if not os.path.exists(self.basic_path):
            print('地址文件夹未找到，请检查')
            sys.exit()
        if not os.path.exists(self.his_stock_ret_path):
            print('股票净值文件未找到，请检查')
            sys.exit()
        if not os.path.exists(self.his_future_ret_path):
            print('期货净值文件未找到，请检查')
            sys.exit()

            # 计算截止到当日的组合收益
    def _compute_ret(self):

        dat = pd.read_excel(self.pool_path, encoding='gbk')
        dat.set_index(dat.columns[0], inplace=True)
        if '权重' not in dat.columns:
            dat['权重'] = 1/len(dat.index)

        wei = dat[['权重']]

        # 算个股当月截至前一交易日的累计收益
        pct = self._data.changepct_daily
        pct = 1 + pct / 100

        yt = datetime.today().year
        mt = datetime.today().month

        cols = [m for m in pct.columns if m.year == yt and m.month == mt]
        tmp_df = pct[cols]
        tmp_cum = tmp_df.cumprod(axis=1)
        pct_chg = tmp_cum[[tmp_cum.columns[-1]]] - 1

        pct_chg = pct_chg.loc[wei.index, :]
        wei.columns = pct_chg.columns
        ret = (wei * pct_chg).sum()

        return ret

    # 通过累计收益计算净值和收益柱状图
    def _compute(self, dat):
        # 计算净值和月度柱状图
        net = dat / dat.iloc[0, 0]
        net.columns = ['净值']

        # 删除重复的月
        to_del = []
        for i in range(1, len(dat.index)):
            if dat.index[i].month == dat.index[i - 1].month:
                to_del.append(dat.index[i - 1])
        if len(to_del) > 0:
            dat.drop(to_del, axis=0, inplace=True)

        bar = dat / dat.shift(1) - 1
        bar.dropna(inplace=True)
        bar.columns = ['收益率']

        return net, bar

    # 计算历史所有期的净值
    def compute_net_value(self, target='stock', freq='W'):
        if target == 'stock':
            net, bar = self._compute_stock_part(freq)
        if target == 'future':
            net, bar = self._compute_future_part()

        return net, bar

    def _compute_stock_part(self, period='M'):
        ret = self._compute_ret()
        dat = pd.read_excel(self.his_stock_ret_path, encoding='gbk')
        dat.set_index(dat.columns[0], inplace=True)

        if ret.index[0] not in dat.index:
            # 找到上个周期最后一个净值数据
            last_net_value = 1
            # 月频周期
            if period.upper() == 'M':
                for j in range(len(dat.index)-1, -1, -1):
                    if dat.index[j].month != ret.index[0].month:
                        last_net_value = dat.iloc[j, 0]
            # 周频周期
            elif period.upper() == 'W':
                for j in range(len(dat.index) - 1, -1, -1):
                    if dat.index[j].isocalendar()[1] != ret.index[0].isocalendar()[1]:
                        last_net_value = dat.iloc[j, 0]

            tmp = last_net_value * (1 + ret)
            dat = pd.concat([dat, pd.DataFrame({dat.columns[0]: tmp})], axis=0)

            dat.to_excel(self.his_stock_ret_path, encoding='gbk')

        net, bar = self._compute(dat)
        return net, bar

    def _compute_future_part(self):
        # 期货的净值是在表里记录的，读取该表
        dat = pd.read_excel(self.his_future_ret_path, encoding='gbk')
        dat.set_index(dat.columns[0], inplace=True)

        net, bar = self._compute(dat)
        return net, bar

    # 保存计算的结果，用于报告
    def save_table(self, net, bar, tar_name):
        self._plot_bar(bar, tar_name)
        self._plot_net(net, tar_name)

    def _plot_net(self, net, tar_name):

        if tar_name == 'stock':
            f_name = '股票'
        elif tar_name == 'future':
            f_name = '期货'

        fig = plt.figure()
        plt.plot(net)
        plt.xticks(rotation=45)
        plt.show()
        fig.savefig(os.path.join(os.path.join(self.basic_path, f_name + '净值走势图.png')))
        plt.close()

    def _plot_bar(self, bar, tar_name):

        if tar_name == 'stock':
            f_name = '股票'
        elif tar_name == 'future':
            f_name = '期货'

        def to_percent(temp, position):
            return '{:^1.0f}%'.format(100 * temp)

        name_list = [i.strftime("%Y-%m-%d") for i in bar.index]
        v_list = [v[0] for v in bar.values]
        fig = plt.figure()
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.bar(range(len(bar.index)), v_list, fc='darkred', tick_label=name_list, width=0.2)
        plt.show()
        fig.savefig(os.path.join(os.path.join(self.basic_path, f_name + '收益柱状图.png')))
        plt.close()


if '__main__' == __name__:
    follow_portfolio = FollowPortfolio()

    # net_future, bar_future = follow_portfolio.compute_net_value(target='future')
    # follow_portfolio.save_table(net_future, bar_future, tar_name='future')
    net_stock, bar_stock = follow_portfolio.compute_net_value(target='stock')
    follow_portfolio.save_table(net_stock, bar_stock, tar_name='stock')












