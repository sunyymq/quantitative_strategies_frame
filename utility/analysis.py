import os
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from openpyxl import load_workbook
from utility.tool0 import Data
from utility.relate_to_tushare import generate_months_ends
import pandas.tseries.offsets as toffsets
from utility.factor_data_preprocess import add_to_panels, align
from utility.tool3 import adjust_months
from utility.constant import data_dair, root_dair
from utility.portfolio import Portfo
from utility.relate_to_tushare import trade_days


register_matplotlib_converters()
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['figure.figsize'] = (15.0, 6.0)  # 图片尺寸设定（宽 * 高 cm^2)

strategy_index_path = r'D:\私募基金相关\私募基金策略指数'


def bool_2_ones(d_df, use_df=False):
    d_ar = d_df.values
    tmp_a = np.full(d_ar.shape, 0)
    tmp_a[d_ar == True] = 1

    if use_df:
        res = pd.DataFrame(data=tmp_a, index=d_df.index, columns=d_df.columns)
    else:
        res = tmp_a
    return res


# 把私募产品数据的列名设置为 日期、净值，注意频率，一般为周频数据


class Analysis:
    def __init__(self, inputs, frequency, benchmark, rf_rate=0.04):
        # ex_ret 是否需要计算超额收益率
        # self.freq 可以是 'y' 'q' 'm' 'w' 'd'
        self.portfolio = None  # 组合净值每日记录
        self.freq = frequency
        self.rf_rate = rf_rate
        self.benchmark = None
        self.benchmark_name = None
        if isinstance(inputs, str):
            self.load_data_str(inputs)
        elif isinstance(inputs, pd.DataFrame):
            self.load_data_df(inputs)

        self.load_benchmark(benchmark)
        # self.benchmark = benchmark

    def load_data_str(self, path):
        ext = path.split('.')[-1]
        if ext == 'csv':
            dat = pd.read_csv(path, encoding='gbk', engine='python')
        elif ext == 'xlsx':
            dat = pd.read_excel(path, encoding='gbk')
        else:
            msg = f"不支持的文件存储类型：{ext}"
            raise TypeError(msg)

        if '日期' not in self.portfolio.columns or '净值' not in self.portfolio.columns:
            print('日期或净值命名错误')
            raise ValueError

        self.portfolio = dat
        self.portfolio.set_index('日期', inplace=True)
        self.trim_df()

    def load_data_df(self, dat):
        if len(dat.columns) == 2 and '日期' not in dat.columns:
            dat.columns = ['日期', '净值']

        self.portfolio = dat

        if '日期' in self.portfolio.columns:
            self.portfolio.set_index('日期', inplace=True)
        if isinstance(self.portfolio.index[0], (int, np.int64)):
            new_index = []
            for d in self.portfolio.index:
                new_index.append(datetime.strptime(str(d), "%Y%m%d"))

            self.portfolio.index = new_index

        self.portfolio['净值'] = self.portfolio['净值'].apply(lambda x: float(x) if isinstance(x, str) else x)

        self.trim_df()

    # 把净值调整为从1开始
    def trim_df(self):
        if self.portfolio.loc[self.portfolio.index[0], '净值'] != 1.0:
            self.portfolio['净值'] = self.portfolio['净值']/self.portfolio.loc[self.portfolio.index[0], '净值']

    def load_benchmark(self, benchmark_type):

        if '净值' not in benchmark_type.columns:
            if '收盘价' in benchmark_type.columns:
                benchmark_type = benchmark_type.rename({'收盘价': '净值'}, axis=1)
            elif '收盘价(元)' in benchmark_type.columns:
                benchmark_type = benchmark_type.rename({'收盘价(元)': '净值'}, axis=1)
            else:
                print('debug，沒有找到净值且未找到相对应的替换项')
                raise ValueError

        if isinstance(benchmark_type, pd.DataFrame):
            self.benchmark = benchmark_type
            self.benchmark_name = '基准'
        else:
            print('无基准收益')
            return None

        # 基准数据比产品净值数据短，把产品净值数据截断，提取前期的数据。
        # 把前面的截断
        if self.benchmark.index[0] > self.portfolio.index[0]:
            to_del = []
            for d in self.portfolio.index:
                if not (d.year == self.benchmark.index[0].year and d.month == self.benchmark.index[0].month):
                    to_del.append(d)
                else:
                    break

            self.portfolio.drop(to_del, axis=0, inplace=True)

        # 把后面的截断
        if self.benchmark.index[-1] < self.portfolio.index[-1]:
            to_del = []
            for d in range(len(self.portfolio.index) - 1, -1, -1):
                if self.benchmark.index[-1] < self.portfolio.index[d]:
                    to_del.append(self.portfolio.index[d])
                else:
                    break

            self.portfolio.drop(to_del, axis=0, inplace=True)

        self.trim_df()

        # 处理基准的数据, is_inner的意思，是基准数据的日期与策略净值数据的日期相同，切基准的日期包含了净值数据的日期。
        new_index = [i for i in self.benchmark.index if i in self.portfolio.index]
        if len(new_index) == len(self.portfolio.index):
            self.benchmark = self.benchmark.loc[new_index, :]
            self.benchmark['净值'] = self.benchmark['净值'] / self.benchmark.loc[self.benchmark.index[0], '净值']
        else:
            start = self.portfolio.index[0]
            end = self.portfolio.index[-1]
            st_loc = np.argmin(np.abs(self.benchmark.index - start))
            ed_loc = np.argmin(np.abs(self.benchmark.index - end))
            self.benchmark = self.benchmark.iloc[st_loc: ed_loc, :]

    def summary(self):

        bench_return = self._bench_return()  # 基准对应区间的累计收益率
        ann_ret = self._annual_return(None)  # 年化收益
        relative_win_rate = self._winning_rate_over_benchmark()  # 相对基准的月度胜率
        ann_vol = self._annual_vol(None)  # 年化波动
        max_wd = self._max_drawdown(None)  # 最大回撤
        sharpe = self._sharpe_ratio(ann_ret=ann_ret, ann_vol=ann_vol)  # 夏普比率
        ann_excess_ret = self._ann_excess_ret()    # 年化超额收益
        te = self._te()

        recommend = None
        if not ann_excess_ret:
            recommend = '无基准，暂无推荐'
        elif ann_ret > 0 and ann_excess_ret > 0:
            recommend = '建议客户继续持有或追加投资'
        elif ann_ret > 0 > ann_excess_ret > -0.05:
            recommend = '建议客户继续持有'
        elif ann_ret > 0 and ann_excess_ret < -0.05:
            recommend = '建议客户减持'
        elif ann_ret < 0 < ann_excess_ret:
            recommend = '建议客户继续持'
        elif -0.1 < ann_ret < 0 and ann_excess_ret < 0:
            recommend = '建议客户减持'
        elif ann_ret < -0.10 and ann_excess_ret < 0:
            recommend = '建议客户全部赎回'

        try:
            self.portfolio.iloc[-1, 0] / self.portfolio.iloc[0, 0] - 1,
        except Exception as e:
            print('debug')

        summary = {
            '开始日期': self.portfolio.index[0].strftime("%Y-%m-%d"),
            '截至日期': self.portfolio.index[-1].strftime("%Y-%m-%d"),
            '累计收益': self.portfolio.iloc[-1, 0] / self.portfolio.iloc[0, 0] - 1,
            '基准名称': self.benchmark_name,
            '基准对应区间累计收益': bench_return,
            '相对基准月度胜率': relative_win_rate,
            '年度收益': ann_ret,
            '年度波动': ann_vol,
            '最大回撤': max_wd,
            '夏普比率': sharpe,
            '年化超额收益': ann_excess_ret,
            '跟踪误差': te,
            '建议': recommend,
        }
        return pd.Series(summary)

    def _te(self):

        bench_pr = self.benchmark['净值'] / self.benchmark['净值'].shift(1) - 1
        bench_pr = bench_pr.dropna()
        portfolio_pr = self.portfolio['净值'] / self.portfolio['净值'].shift(1) - 1
        portfolio_pr = portfolio_pr.dropna()

        td = portfolio_pr - bench_pr
        te = np.sqrt( np.sum( (td - np.mean(td)) ** 2 ) / (len(td) - 1) )

        if self.freq.lower() == 'm':
            te = np.sqrt(min(len(td), 12)) * te
        elif self.freq.lower() == 'd':
            te = np.sqrt(min(len(td), 250)) * te
        elif self.freq.lower() == 'q':
            te = np.sqrt(min(len(td), 4)) * te
        elif self.freq.lower() == 'odd':
            num_in_year = []
            for y in range(2015, 2019):
                num_in_year.append(len([i for i in td.index if i.year == y]))
            lens_mean = np.mean(num_in_year)
            te = np.sqrt(min(len(td), lens_mean)) * te

        return te

    def return_each_month(self):

        bench_pr = self.benchmark['净值'] / self.benchmark['净值'].shift(1) - 1
        bench_pr = bench_pr.dropna()
        portfolio_pr = self.portfolio['净值'] / self.portfolio['净值'].shift(1) - 1
        portfolio_pr = portfolio_pr.dropna()

        td = portfolio_pr - bench_pr

        month_ret = pd.DataFrame({'组合收益': portfolio_pr, '基准收益': bench_pr, '月度超额': td})

        return month_ret

    def return_each_year(self):
        st_y = self.portfolio.index[0].year
        ed_y = self.portfolio.index[-1].year

        res = pd.DataFrame(index=range(st_y, ed_y + 1, 1), columns=['年度收益', '最大回撤'])
        for y in range(st_y, ed_y + 1, 1):
            tmp = self.portfolio.loc[self.portfolio.index[self.portfolio.index.year == y], :]
            tmp_start = tmp.index[0]
            tmp_end = tmp.index[-1]
            max_wd = self._max_drawdown(acc_rets=None, start_date=tmp_start, end_date=tmp_end)    # 最大回撤

            res.loc[y, '年度收益'] = tmp.loc[tmp.index[-1], '净值'] / tmp.loc[tmp.index[0], '净值'] - 1
            res.loc[y, '最大回撤'] = max_wd
            if isinstance(self.benchmark, pd.DataFrame):
                if tmp_start in self.benchmark.index and tmp_end in self.benchmark.index:
                    res.loc[y, '基准收益'] = self.benchmark.loc[tmp_end, '净值'] / \
                                                     self.benchmark.loc[tmp_start, '净值'] - 1

        res.index.name = '年份'
        return res

    def plot_pic(self, save_name):

        tmp = pd.DataFrame({'产品净值': self.portfolio['净值'], '基准净值': self.benchmark['净值']})
        tmp.dropna(how='any', inplace=True)
        plt.plot(tmp)
        # plt.title(fname)
        plt.legend(tmp.columns, loc=0)
        plt.savefig(os.path.join(r'D:\私募基金相关\pic', save_name + f'.png'))
        plt.close()

        return None

    def _get_date_gap(self, freq='d'):

        start_date = self.portfolio.index[0]
        end_date = self.portfolio.index[-1]
        days = (end_date - start_date) / toffsets.timedelta(1)

        return days

    def _annual_return(self, net_vals=None):

        if net_vals is None:
            net_vals = self.portfolio['净值']
            start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        else:
            start_date, end_date = net_vals.index[0], net_vals.index[-1]

        tmp = list(net_vals.index)
        start_date_id = tmp.index(start_date)
        end_date_id = tmp.index(end_date)
        if start_date_id == 0:
            net_vals = net_vals[start_date_id:end_date_id + 1]
            # 净值从1开始
            total_ret = net_vals.values[-1] / net_vals.values[0] - 1
        else:
            # 对于月频数据，计算收益从start_date_id前一个
            net_vals = net_vals[start_date_id - 1:end_date_id + 1]
            total_ret = net_vals.values[-1] / net_vals.values[0] - 1

        date_gap = self._get_date_gap()
        exp = 365 / date_gap
        ann_ret = (1 + total_ret) ** exp - 1

        return ann_ret

    def _annual_vol(self, net_vals=None):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        if 'netval_pctchg' not in self.portfolio.columns:
            self.portfolio['netval_pctchg'] = self.portfolio['净值'] / self.portfolio['净值'].shift(1)

        if net_vals is None:
            ret_per_period = self.portfolio['netval_pctchg']

        ret_per_period = ret_per_period.loc[start_date:end_date]
        # 年化波动率 = 日频收益率标准差 * sqrt(250)
        # 年化波动率 = 周频收益率标准差 * sqrt(52)
        # 年化波动率 = 月频收益率标准差 * sqrt(12)
        ret_per_period = ret_per_period.dropna()
        if self.freq.lower() == 'y':
            ann_vol = ret_per_period.std()
        elif self.freq.lower() == 'q':
            ann_vol = ret_per_period.std() * np.sqrt(4)
        elif self.freq.lower() == 'm':
            ann_vol = ret_per_period.std() * np.sqrt(12)
        elif self.freq.lower() == 'w':
            ann_vol = ret_per_period.std() * np.sqrt(52)
        elif self.freq.lower() == 'd':
            ann_vol = ret_per_period.std() * np.sqrt(250)
        elif self.freq.lower() == 'odd':
            # 不规律的调仓周期，先计算年内的调仓次数
            num_in_year = []
            for y in range(2015, 2019):
                num_in_year.append(len([i for i in ret_per_period.index if i.year == y]))
            lens_mean = np.mean(num_in_year)
            ann_vol = ret_per_period.std() * np.sqrt(lens_mean)

        return ann_vol

    def _max_drawdown(self, acc_rets=None, start_date=None, end_date=None):

        if not start_date and not end_date:
            start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]

        if acc_rets is None:
            acc_rets = self.portfolio['净值'] - 1

        acc_rets = acc_rets.loc[start_date:end_date]
        max_drawdown = (1 - (1 + acc_rets) / (1 + acc_rets.expanding().max())).max()
        return max_drawdown

    def _sharpe_ratio(self, ann_ret=None, ann_vol=None):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        if ann_ret is None:
            ann_ret = self._annual_return(start_date, end_date)
        if ann_vol is None:
            ann_vol = self._annual_vol(start_date, end_date)
        return (ann_ret - self.rf_rate) / ann_vol

    def form_daily_return(self, net_value):
        # net_value = self.portfolio
        res = net_value['净值'] / net_value['净值'].shift(1) - 1
        res.fillna(0, inplace=True)
        return res

    def _get_excess_acc_ret(self):

        if not isinstance(self.benchmark, pd.DataFrame):
            return None

        dr_s = self.form_daily_return(self.portfolio)
        dr_b = self.form_daily_return(self.benchmark)

        dr_e = dr_s - dr_b
        res = (dr_e+1).cumprod()

        return res

    def _ann_excess_ret(self):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        if isinstance(self.benchmark, pd.DataFrame):
            # if len(self.benchmark.index) == len(self.portfolio.index):
            #     excess_acc_ret = self._get_excess_acc_ret()
            #     ann_excess_ret = self._annual_return(net_vals=excess_acc_ret)
            #     return ann_excess_ret
            # else:
            bench_return = self.benchmark.loc[self.benchmark.index[-1], '净值'] / \
                           self.benchmark.loc[self.benchmark.index[0], '净值'] \
                           - 1
            portfolio_return = self.portfolio.loc[self.portfolio.index[-1], '净值'] / \
                               self.portfolio.loc[self.portfolio.index[0], '净值'] \
                               - 1
            ann_excess_ret = portfolio_return - bench_return

            days = np.min([(self.portfolio.index[-1] - self.portfolio.index[0]).days,
                          (self.benchmark.index[-1] - self.benchmark.index[0]).days])

            exp = 365 / days
            ann_excess_ret = (1 + ann_excess_ret) ** exp - 1

            return ann_excess_ret
        else:
            return None

    # def _get_excess_acc_ret(self, start_date=None, end_date=None):
    #
    #     bm_ret = self.portfolio_record['benchmark']
    #     nv_ret = self.portfolio_record['netval_pctchg']
    #
    #     if start_date and end_date:
    #         bm_ret = bm_ret.loc[start_date:end_date]
    #         nv_ret = nv_ret.loc[start_date:end_date]
    #     excess_ret = nv_ret.values.flatten() - bm_ret.values.flatten()
    #     excess_acc_ret = pd.Series(np.cumprod(1+excess_ret), index=nv_ret.index)
    #     return excess_acc_ret

    def _winning_rate(self):

        start_date, end_date = self.portfolio.index[0], self.portfolio.index[-1]
        nv_pctchg = self.portfolio['netval_pctchg']

        if start_date and end_date:
            nv_pctchg = nv_pctchg.loc[start_date:end_date]

        nv_pctchg = nv_pctchg.dropna()
        win = (nv_pctchg > 1)
        win_rate = np.sum(win) / len(win)
        return win_rate

    def _winning_rate_over_benchmark(self):
        if isinstance(self.benchmark, pd.DataFrame):
            if len(self.benchmark.index) == len(self.portfolio.index):
                pf = self.portfolio[self.portfolio.columns[0]]
                bench = self.benchmark[self.benchmark.columns[0]]
                pf_r = pf / pf.shift(1) - 1
                pf_r = pf_r.dropna()
                bench_r = bench / bench.shift(1) - 1
                bench_r = bench_r.dropna()
                try:
                    win_or_not = pf_r > bench_r
                except Exception as e:
                    print('deby')
                win_rate = np.sum(win_or_not) / len(win_or_not)
                return win_rate
            else:
                return None
        else:
            return None

    def _bench_return(self):  # 基准对应区间的累计收益率

        if not isinstance(self.benchmark, pd.DataFrame):
            return None

        bench_return = self.benchmark.loc[self.benchmark.index[-1], '净值'] / self.benchmark.loc[self.benchmark.index[0], '净值'] \
                       - 1

        return bench_return


# 回测类：逻辑上仅处理月频调仓的处理，对于需要用到日频率数据的freq设为D，不需要的freq设为M，暂不用涉及到季度调仓的策略。
class BackTest:
    def __init__(self, wei, freq, fee_type='fee', benchmark_str='WindA',
                 hedge_status=True, hedge_para_dict={}):
        '''
        :param wei:          下个月需要配置的股票权重，在月末计算出来，比如7月31日算出下个月应该配置的股票权重
        :param freq:         调仓频率，'M'是月频，'Odd'是季度频率或其他大于月频的调仓频率，需和wei保持一致
        :param fee_type:
        :param benchmark_str:
        :param hedge_status:
        :param hedge_para_dict:
        '''

        self.data = Data()
        self.weight = wei                        # 股票权重
        self.freq = freq                         # 处理类型：频率
        # 对一些奇异月份换仓频率的情况
        self.month_num_dict = None
        if self.freq.upper() in ['ODD', 'Q']:
            self.decide_month_num()

        self.changePCT_np = None                 # 价格变动百分比
        self.market_value = None                 # 市值
        self.net_value = None                    # 净值
        self.net_pct = None
        self.net_pct_nofee = None
        self.net_value_nofee = None
        self.fee_type = fee_type
        self.open_price = None
        self.close_price = None
        self.load_stock_price()
        if self.fee_type == 'fee':
            self.portfo = Portfo(self.open_price,  self.close_price)
            self.real_wei = None

        self.load_pct()
        self.benchmark_str = None
        self.benchmark_p = None                  # 基准指数的表现
        self.benchmark_r = None
        self.load_benchmark(benchmark_str)

        self.fee_total = 0
        self.impact_cost = 0

        self.summary = None                      # 策略评价指标
        # self.hedge_status = hedge_status
        # if self.hedge_status:
        #     self.hedging_method = None           # 市值对冲还是beta对冲
        #     self.stock_beta = None               # 股票的beta系数，通过读取因子矩阵获得
        #     self.future_beta = None              # 期货合约的beta系数，通过读取矩阵获得
        #     self.hedging_rate = None             # 对冲比例
        #
        # # 如果是beta对冲，需要导入股票和期货合约的beta值
        # if self.hedge_status and self.hedging_method == 'beta':
        #     self.load_beta_value()

    def decide_month_num(self):
        # 先确定每个调仓区间的月份数量
        month_num_dict = {}  # key为开始月末，value为月的数量
        for i in range(1, len(self.weight.columns)):
            peri_i = i - 1
            m_now = self.weight.columns[i].month
            m_peri = self.weight.columns[peri_i].month
            if m_now < m_peri:
                mn = 12 + m_now - m_peri
            else:
                mn = m_now - m_peri
            month_num_dict.update({self.weight.columns[peri_i]: mn})

        self.month_num_dict = month_num_dict

    def load_benchmark(self, benchmark_str):

        if not benchmark_str:
            benchmark_str = 'WindA'

        self.benchmark_str = benchmark_str + '指数'
        if self.freq.upper() == 'M':
            if benchmark_str in ['WindA', 'HS300', 'SH50', 'ZZ500']:
                price_monthly = self.data.index_price_monthly
                self.benchmark_p = price_monthly.loc[benchmark_str, :]
                self.benchmark_r = self.benchmark_p / self.benchmark_p.shift(1) - 1
            else:
                price_monthly = self.data.industry_price_monthly
                self.benchmark_p = price_monthly.loc[benchmark_str + '（申万）', :]
                self.benchmark_r = self.benchmark_p / self.benchmark_p.shift(1) - 1
        elif self.freq.upper() in ['ODD', 'Q']:

            # 先确定一个月频的基本数据
            if benchmark_str in ['WindA', 'HS300', 'SH50', 'ZZ500']:
                price_monthly = self.data.index_price_monthly
                benchmark_p = price_monthly.loc[benchmark_str, :]
            else:
                price_monthly = self.data.industry_price_monthly
                benchmark_p = price_monthly.loc[benchmark_str + '（申万）', :]

            months = list(self.month_num_dict.keys())
            self.benchmark_p = benchmark_p[months]
            self.benchmark_r = self.benchmark_p / self.benchmark_p.shift(1) - 1

        elif self.freq.upper() == 'W':
            if benchmark_str not in ['WindA', 'HS300', 'SH50', 'ZZ500']:
                print('相应的基准无周频数据，基准变更为WindA')
                benchmark_str = 'WindA'

            if benchmark_str in ['WindA', 'HS300', 'SH50', 'ZZ500']:
                price_weekly = self.data.index_price_daily.T
                tds_w = trade_days(freq='w')
                tds_w = [d for d in tds_w if d in price_weekly.index]
                self.benchmark_p = price_weekly.loc[tds_w, benchmark_str]
                self.benchmark_r = self.benchmark_p / self.benchmark_p.shift(1) - 1

    def load_stock_price(self):
        open = self.data.openprice_daily
        close = self.data.closeprice_daily
        adj = self.data.adjfactor
        open_aj = open * adj
        close_aj = close * adj
        self.open_price = open_aj
        self.close_price = close_aj

    # 自然载入变动百分比数据
    def load_pct(self):
        if self.freq.upper() == 'M':         # 导入月度价格变动百分比数据
            self.changePCT_np = self.data.PCT_CHG_NM / 100

        elif self.freq.upper() in ['ODD', 'Q']:
            # 再得到相应的价格变动幅度
            changePCT_next_month = self.data.PCT_CHG_NM / 100

            self.changePCT_np = pd.DataFrame()
            for k, v in self.month_num_dict.items():
                # 找出月份区间再计算累乘值
                loc = np.where(changePCT_next_month.columns == k)[0][0]
                tmp = changePCT_next_month.iloc[:, loc:loc+v]
                tmp = tmp + 1
                tmp = tmp.cumprod(axis=1) - 1

                tmp_df = pd.DataFrame({k: tmp[tmp.columns[-1]]})
                self.changePCT_np = pd.concat([self.changePCT_np, tmp_df], axis=1)

        elif self.freq.upper() == 'W':
            self.changePCT_np = self.data.PCT_CHG_NW / 100

    def load_beta_value(self):
        # 未加预测，直接使用截面日期的数据
        p = os.path.join(root_dair, '单因子检验', '因子矩阵', 'beta.csv')
        stock_beta = pd.read_csv(p, encoding='gbk')
        stock_beta.set_index(stock_beta.columns[0], inplace=True)
        stock_beta.columns = pd.to_datetime(stock_beta.columns)
        self.stock_beta = self.stock_beta
        self.future_beta = self.data.sf_beta

    # 买入目标股票到一定比例
    def order_percent(self, code, percent, dt):
        self.portfo.order(code, percent)

    # 程序逻辑： 对于月频率数据，如果月频调仓，那么就比较简单，知道权重和change相乘就可以。
    def run_bt(self):

        # 先对行和列取交集，然后再转换为array，再矩阵乘法得到每个月的收益，再判断是不是要去费，
        [self.weight, self.changePCT_np] = align(self.weight, self.changePCT_np)
        self.weight.fillna(0, inplace=True)
        self.changePCT_np.fillna(0, inplace=True)

        if self.fee_type == 'No_fee':
            ret_df = self.weight * self.changePCT_np

            # tt = self.weight[pd.to_datetime(datetime(2020, 8, 31))]
            # tt[tt > 0]

            ret_df.sum()
            # 月度收益pct
            self.net_pct = ret_df.sum(axis=0)
            self.net_value = (self.net_pct + 1).cumprod()

            # 因权重和价格均为为下一期的数据，所以净值需向下移动一期
            self.net_pct = self.net_pct.shift(1)
            self.net_value = self.net_value.shift(1)
            self.net_value.dropna(inplace=True)

        else:
            ret_df = self.weight * self.changePCT_np
            # 无费用和冲击成本的收益，可用来对比使用
            self.net_pct_nofee = ret_df.sum(axis=0)
            self.net_value_nofee = (self.net_pct_nofee + 1).cumprod()
            self.net_value_nofee = self.net_value_nofee.shift(1).dropna()

            h, l = self.weight.shape
            # 按列循环，并先选出需要卖出的，按照先卖出后买入的顺序来处理。
            for l_loc in range(0, l-1):
                # if l_loc == 2:
                #     print('deg')
                dt = self.weight.columns[l_loc]
                loc = np.where(dt == self.open_price.columns)[0][0]
                # 权重日期的后一个日期是开盘价格的真实日期
                real_dt = self.open_price.columns[loc+1]
                print('权重日期为{}，真实日期为{}'.format(dt, real_dt))

                try:
                    price = self.open_price[real_dt]
                    # 用开盘价设置portfo类的价格和日期
                    if real_dt.month == 5 and real_dt.year == 2011:
                        print('here')
                    self.portfo.update_price_and_dt(price, real_dt, 'open')
                    self.portfo.update()
                    err = self.portfo.self_check()
                    if abs(err) > 100:
                        print('误差过大')

                    # self.real_wei是经过持有期后组合的实际股票权重，对于月频调仓，就是月末的股票实际权重。real_wei初始状态为None
                    # 计算change的目的主要是【排序】，保证交易股票的顺序是先操作需要卖出的股票，再操作需要买入的股票
                    if isinstance(self.real_wei, pd.Series):
                        weight_e = self.weight[dt]
                        both = pd.DataFrame({'real': self.real_wei,
                                             'e': weight_e}).fillna(0.0)
                        change = both['real'] - both['e']
                        change.sort_values(ascending=False, inplace=True)
                    else:
                        # 第一个月，开始建仓
                        change = self.weight[dt]

                    for h_loc in range(0, h):
                        code = change.index[h_loc]
                        tmp_wei = self.weight.loc[code, dt]

                        if real_dt == datetime(2011, 5, 3) and code == '600572.SH':
                            print('debug')

                        tmp_h = np.where(self.weight.index == code)[0][0]
                        try:
                            pre_wei = self.weight.iloc[tmp_h, l_loc - 1]
                        except Exception as e:
                            pre_wei = 0

                        need_to_deal = False
                        # 先判断该股票是否需要操作
                        if l_loc == 0:
                            # 第一期
                            if tmp_wei == 0:
                                need_to_deal = False
                            else:
                                need_to_deal = True
                        else:
                            # 上一期及当期该股票权重都为0，无需操作
                            if tmp_wei == 0 and pre_wei == 0:
                                need_to_deal = False
                            elif isinstance(self.real_wei, pd.Series) and code in self.real_wei.index \
                                    and self.real_wei[code] == tmp_wei:
                                need_to_deal = False
                            else:
                                need_to_deal = True

                        if need_to_deal:
                            print('{}，{}，{}'.format(real_dt, code, tmp_wei))
                            self.portfo.order(code, tmp_wei)
                            err = self.portfo.self_check()
                            print('err={}'.format(err))
                            if abs(err) > 100:
                                print('误差过大')
                                self.portfo.self_check()
                        if np.any(pd.isna(self.portfo.hold_assets)):
                            print('出现nan')

                    # 在月初开盘买卖阶段，加上佣金和税费，不能有太大的误差
                    err = self.portfo.self_check()
                    if abs(err) > 100:
                        print('误差过大')
                        self.portfo.self_check()

                except Exception as e:
                    print('debug')

                # 这个持仓周期过完，得到最新的收盘价
                real_dt = self.weight.columns[l_loc+1]
                price = self.close_price[real_dt]

                # 根据新的收盘价，计算权重、记录组合总市值
                self.portfo.update_price_and_dt(price, real_dt, 'close')
                self.portfo.update(recoding=True)
                err = self.portfo.self_check(fee_isin=False)
                if abs(err) > 100:
                    print('误差过大')
                self.real_wei = self.portfo.get_stocks_wei()

            self.market_value = self.portfo.his_asset
            self.net_value = self.market_value / self.portfo.initial_asset
            plt.plot(self.net_value)
            plt.plot(self.net_value_nofee)
            plt.legend(['扣费', '不扣费'])
            plt.show()
            # todo 对冲的话，还是把股指日频的处理后转换成月频的来算，股票如果也用日频的，太麻烦。
            if self.hedge_status:
                pass

            return self.net_value


    def hedging_fun(self):
        pass

    def analysis(self):
        dat_df = pd.DataFrame({'净值': self.net_value})
        dat_df.index.name = '日期'
        bm_df = pd.DataFrame({'净值': self.benchmark_p})
        ana = Analysis(dat_df, self.freq.lower(), benchmark=bm_df)
        summary = ana.summary()
        tr = self.turnover_rate()
        summary = summary.append(pd.Series(index=['换手率'], data=[tr]))
        summary = pd.DataFrame({'评价指标': summary})
        self.summary = summary

        each_year = ana.return_each_year()
        each_month = ana.return_each_month()

        return summary, each_year, each_month

    # 计算换手率
    def turnover_rate(self):
        wei_change = np.abs(self.weight - self.weight.shift(1, axis=1).fillna(0))
        turnover_rate = np.sum(wei_change / 2, axis=0)
        turnover_rate = np.mean(turnover_rate) * 12
        return turnover_rate

    def plt_pic(self, show=False):
        pic_df = pd.DataFrame({'组合净值': self.net_value, self.benchmark_str: self.benchmark_p})
        pic_df.dropna(how='any', inplace=True)
        pic_df = pic_df/pic_df.iloc[0, :]
        fig = plt.figure()
        plt.plot(pic_df)
        plt.legend(pic_df.columns)
        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    try:
        stock_wt = pd.read_csv(r'D:\Database_Stock\临时\权重.csv', encoding='gbk')
        stock_wt.set_index(stock_wt.columns[0], inplace=True)
        stock_wt.columns = pd.to_datetime(stock_wt.columns)

        bt = BackTest(stock_wt, 'Odd', fee_type='fee')
        bt.run_bt()
        bt.plt_pic()
        print(bt.analysis())

    except Exception as e:
        print('权重文件未找到，请正确配置权重文件地址')



