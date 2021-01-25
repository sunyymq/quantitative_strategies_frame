# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:25:00 2019

@author: HP
"""
from utility.tool0 import Data
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.tseries.offsets as toffsets
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from copy import deepcopy
from utility.constant import factor_path, sf_test_save_path, work_dir, info_cols, total_result_path

warnings.filterwarnings('ignore')  # 将运行中的警告信息设置为“忽略”，从而不在控制台显示

total_result_dict = {}

industry_benchmark = 'zx'     # 行业基准-中信一级行业

plt.rcParams['font.sans-serif'] = ['SimHei']   # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
plt.rcParams['figure.figsize'] = (15.0, 6.0)   # 图片尺寸设定（宽 * 高 cm^2)
plt.rcParams['font.size'] = 10                 # 字体大小
num_layers = 2                                 # 设置分层层数
tick_spacing1 = 12                             # 设置画图的横轴密度
tick_spacing2 = 150                            # 设置横坐标密度


# --------------------------------
# 根据所有ic值计算icir值
def compute_ir(dat_df, n):
    res = pd.DataFrame()
    # 计算ICIR
    for col, val in dat_df.iteritems():
        tmp = pd.Series(index=val.index)
        for i in range(n, len(val)):
            tmp[tmp.index[i - 1]] = np.nanmean(val[val.index[i - n:i]]) / np.nanstd(val[val.index[i - n:i]])

        tmp.drop(tmp.index[:n - 1], inplace=True)
        res = pd.concat([res, pd.DataFrame({col: tmp})], axis=1)

    res.index.name = 'date'
    return res


# ------------------------
# 得到因子名称
def get_factor_names(factor_p):
    global info_cols
    dir_list = os.listdir(factor_p)
    # 使用最后一个月的列名，因为前期的可能因为行业数据较少，而短缺。
    panel_f = dir_list[-1]
    panel_pd = pd.read_csv(os.path.join(factor_p, panel_f),
                            encoding='gbk', engine='python')

    factor_names = [col for col in panel_pd.columns]
    info_cols_tmp = [col for col in info_cols if col in panel_pd.columns]
    factor_names = list(set(factor_names) - set(info_cols_tmp))

    return factor_names



# ------------------------
# 加权回归
def regress(y, X, w=1, intercept=False):
    if intercept:
        X = sm.add_constant(X)
    try:
        all = pd.concat([y, X], axis=1)
        all = all.dropna()
        y = all[all.columns[0]]
        X = all[all.columns[1:]]
        w = w[y.index]
        model = sm.WLS(y, X, weights=w)
        result = model.fit()
    except  Exception as e:
        a = 2

    ts, params = result.tvalues, result.params
    ts.index = X.columns
    params.index = X.columns
    resid = y - np.dot(X, params.T)   
    return ts, params, resid


# ------------------------
# 返回 市值和 行业哑变量组成的矩阵
def get_ind_mktcap_matrix(datdf, ind=True, mktcap=True):

    if mktcap:
        lncap = np.log(datdf['Mkt_cap_float'])
        lncap.name = 'ln_mkt_cap'
    else:
        lncap = pd.DataFrame()
    if ind:
        ind_dummy_matrix = pd.get_dummies(datdf[f'Industry_sw'])
    else:
        ind_dummy_matrix = pd.DataFrame()
    
    return pd.concat([lncap, ind_dummy_matrix], axis=1)


# ------------------------
# 得到ic值
def get_ic(datdf, fac_name, neutralize=False):
    pctchgnm = datdf['Pct_chg_nm']
    facdat = datdf[fac_name]
    if neutralize:
        ind_mktcap_matrix = get_ind_mktcap_matrix(facdat)
        _, _, facdat = regress(facdat, ind_mktcap_matrix)
    dat = pd.concat([facdat, pctchgnm], axis=1)
    ic = dat.corr().iat[0, 1]
    return ic


# ------------------------
# 回归参数总结
def regression_summary(ts, params, ics):
    res = {}
    res['t值绝对值平均值'] = np.nanmean(np.abs(ts))                  # t值绝对值平均值
    res['t值绝对值>2概率'] = len(ts[np.abs(ts) > 2]) / len(ts)    # t值绝对值>2概率
     
    res['因子收益平均值'] = np.nanmean(params)                       # 因子收益平均值
    res['因子收益标准差'] = np.nanstd(params)                        # 因子收益标准差
    res['因子收益t值'] = stats.ttest_1samp(params[~pd.isnull(params)], 0).statistic     # 因子收益t值
    res['因子收益>0概率'] = len(params[params > 0]) / len(params)   # 因子收益>0概率
    
    res['IC平均值'] = np.nanmean(ics)                                # IC平均值
    res['IC标准差'] = np.nanstd(ics)                                 # IC标准差
    res['IRIC'] = res['IC平均值'] / res['IC标准差']               # IRIC
    res['IC>0概率'] = len(ics[ics>0]) / len(ics)                  # IC>0概率
    return pd.Series(res)


# ------------------------
# 单一factor的t值、ic值检验， 输入是 datpanel 格式的
def t_ic_test(datpanel, factor_name, add_ind=True):
    t_series, fret_series, ic_series = pd.Series(), pd.Series(), pd.Series()
    for date, datdf in datpanel.items():
        # print(date)
        w = np.sqrt(datdf['Mkt_cap_float'])
        y = datdf['Pct_chg_nm']
        if factor_name not in datdf.columns:
            continue

        try:
            X = datdf[[factor_name]]
        except Exception as e:
            print('bug')

        if len(X.dropna()) == 0 or len(y.dropna()) == 0:
            continue

        if add_ind:
            ind_mktcap_matrix = get_ind_mktcap_matrix(datdf)
            X = pd.concat([X, ind_mktcap_matrix], axis=1)

        ts, f_rets, _= regress(y, X, w)
        
        t_series[date] = ts[factor_name]
        fret_series[date] = f_rets[factor_name]

        ic = get_ic(datdf, factor_name)
        ic_series[date] = ic

    if len(t_series) == 0:
        summary = pd.Series()
    else:
        summary = regression_summary(t_series.values, fret_series.values,
                                     ic_series.values)

    return [summary, t_series, fret_series, ic_series]


# ------------------------
def get_datdf_in_panel(factor_path, factor_names=None):

    dates = []
    for f in os.listdir(factor_path):
        curdate = f.split('.')[0]
        dates.append(curdate)

    # 读取相应月份得数据
    datpanel = {}
    for date in dates:     
        datdf = pd.read_csv(os.path.join(factor_path, date+'.csv'),
                            engine='python', encoding='gbk', index_col=[0])
        if 'Code' in datdf.columns:
            datdf.set_index('Code', inplace=True)

        date = pd.to_datetime(date)
        if not factor_names:
            datpanel[date] = datdf

        elif factor_names:
            new_cols = [fn for fn in factor_names if fn in datdf.columns]
            datpanel[date] = datdf[new_cols]

    return datpanel


# ------------------------
# factor list的t值、ic值检验， 输入是 datpanel 格式的
def get_test_result(factors, datpanel, ind=True):
    res = pd.DataFrame()
    ts_all, frets_all, ics_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for factor_name in factors:
        # print(factor_name)
        cur_fac_res, ts, frets, ics = t_ic_test(datpanel, factor_name, add_ind=ind)
        col_name = factor_name.replace('/','_div_') if '/' in factor_name else factor_name
        
        cur_fac_res.name = col_name
        ts.name = col_name
        frets.name = col_name
        ics.name = col_name
        
        res = pd.concat([res, cur_fac_res], axis=1)
        ts_all = pd.concat([ts_all, ts], axis=1)
        frets_all = pd.concat([frets_all, frets], axis=1)
        ics_all = pd.concat([ics_all, ics], axis=1)
        
    ts_all = ts_all.sort_index()
    frets_all = frets_all.sort_index()
    ics_all = ics_all.sort_index()
    return res, ts_all, frets_all, ics_all


# ------------------------
# 按年度遍历来计算factors 的t值和ic值
def single_factor_test(path_dict, factors=None, icir_window=12, ind=True):
    '''
    增加了分板块测试的功能，plate是一个tuple,0为板块名称，1为其包含得一级行业，若是单行业测试，则其名称和一级行业相同
    '''

    global info_cols
    print("\n开始进行T检验和IC检验...")

    sf_test_save_path = path_dict['sf_test_save_path']
    factor_path = path_dict['factor_path']

    datpanel = get_datdf_in_panel(factor_path)
    # 若未指定特定测试的因子，则默认测试所有因子
    if factors is None:
        factors = get_factor_names()
    test_result, ts, frets, ics = get_test_result(factors, datpanel, ind=ind)

    icir = compute_ir(ics, icir_window)

    # 存储所有t值、因子收益率、ic值时间序列数据
    for save_name, df in zip(['t_value', 'factor_return', 'ic', 'icir', 'T检验&IC检验结果'],
                             [ts, frets, ics, icir, test_result]):
        df.to_csv(os.path.join(sf_test_save_path, save_name+'.csv'),
                  encoding='gbk')

    # 绘制单因子检验图，并进行存储
    plot_test_figure(ts, frets, ics, save=True)
    print(f"检验完毕！结果见目录：{path_dict['sf_test_save_path']}")
    print('*'*80)

# ------------------------
# 画图输出
def plot_test_figure(ts, frets, ics, save=True, plate=None):
    global sf_test_save_path, industry_factor_path
    ts = np.abs(ts)
    factors = ts.columns
    fig_save_path = os.path.join(sf_test_save_path, 'T检验与IC检验结果图')
    if not os.path.exists(fig_save_path):
        os.mkdir(fig_save_path)
    for fac in factors:
        t, fret, ic = ts[fac], frets[fac], ics[fac]
        sharedx = [str(d)[:10] for d in t.index]
        
        fig, axes = plt.subplots(3, 1, sharex=True)
        fig.suptitle(fac)
        bar_plot(axes[0], sharedx, t.values, 't value绝对值')
        bar_plot(axes[1], sharedx, fret.values, '因子收益率')
        bar_plot(axes[2], sharedx, ic.values, 'IC')
        if plate:
            fig.savefig(os.path.join(industry_factor_path, plate, '单因子检验', fac + '.png'),
                        encoding='gbk')
        else:
            postfix = '\\' + fac + '.png'
            fig.savefig(fig_save_path + postfix)
        plt.close()


# ------------------------
# 画柱状图
def bar_plot(ax, x, y, title):
    global tick_spacing1
    ax.bar(x, y)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing1))
    # 设置x轴坐标字符大小
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
    # 设定title的字符大小
    ax.set_title(title, fontdict={'fontsize': 15})


# # ------------------------
# # 得到所有行业的列表
# def get_industry(industry_benchmark='zx'):
#     global factor_path
#
#     benchmark = 'industry_' + industry_benchmark
#
#     f = os.listdir(factor_path)
#     f = f[-1]
#     datdf = pd.read_csv(os.path.join(factor_path, f),
#                         engine='python', encoding='gbk', index_col=[0])
#
#     return datdf[benchmark].drop_duplicates().tolist()


def get_firt_industry_list():
    data = Data()
    stock_basic = data.stock_basic_inform
    industry_names = list(set(stock_basic['申万一级行业'].values))

    # 有些公司刚上市，还没有分类，就会有nan的情况
    ret = []
    for ina in industry_names:
        if isinstance(ina, str):
            ret = ret + [ina]

    return ret


### 净值回测
class Backtest_stock:
    def __init__(self, *, market_data, start_date, end_date, benchmarkdata=None,
                 stock_weights=None, initial_capital=100000000, tradedays=None, 
                 refreshdays=None, trade_signal=None, stk_basic_data=None, 
                 rf_rate=0.04, use_pctchg=False, **kwargs):
        if stock_weights is None:
            if trade_signal is None:
                raise AttributeError("PARAM::stock_weights must be passed in.")
                # 证券权重数据和交易信号均为空时，报错（两者至少传入其一）

        self.use_pctchg = use_pctchg                  # 是否采用pctchg进行回测净值计算
        if stk_basic_data is None:
            self.stock_pool = stock_weights.index    # 股票池
        else:
            self.stock_pool = stk_basic_data.index
            
        self.stock_weights = stock_weights           # 各组的证券权重
        self.market_data = market_data               # 对于二级行业测试，需要传入相关的行业数据
        # 行情数据（全A股复权收盘价 或 A股日涨跌幅）
        self.benchmark_data = benchmarkdata          # 基准（000300或000905日涨跌幅）
        
        self.start_date = start_date                 # 回测开始日期
        self.end_date = end_date                     # 回测结束日期
        self.capital = initial_capital               # 可用资金
        self.net_value = initial_capital             # 账户市值
        
        self.curdate = None                          # 当前调仓交易日对应日期
        self.lstdate = None                          # 上一个调仓交易日对应日期
        
        if tradedays:                                # 回测期内所有交易日list
            tradedays = pd.to_datetime(tradedays)
        else:
            tradedays = pd.to_datetime(self.market_data.columns)
        self.tradedays = sorted(tradedays)

        # 回测期内所有调仓交易日list, 对于月频数据测试，需要自己传入regreshdays，与月度因子数据同日期
        self.refreshdays = refreshdays
        
        self.position_record = {}                    # 每个交易日持仓记录
        self.portfolio_record = {}                   # 组合净值每日记录
        self.rf_rate = rf_rate

    def _get_date_idx(self, date):
        """
        返回传入的交易日对应在全部交易日列表中的下标索引
        """
        datelist = list(self.tradedays)
        date = pd.to_datetime(date)
        try:
            idx = datelist.index(date)
        except ValueError:
            datelist.append(date)
            datelist.sort()
            idx = datelist.index(date)
            if idx == 0:
                return idx + 1
            else:
                return idx - 1
        return idx

    def run_backtest(self):
        """
        回测主函数
        """
        start_idx = self._get_date_idx(self.start_date)
        end_idx = self._get_date_idx(self.end_date)
        
        hold = False
        for date in self.tradedays[start_idx:end_idx+1]:
            # 对回测期内全部交易日遍历，每日更新净值
            if date in self.refreshdays:
                # 如果当日为调仓交易日，则进行调仓
                hold = True
                idx = self.refreshdays.index(date)
                if idx == 0:
                    # 首个调仓交易日
                    self.curdate = date
                self.lstdate, self.curdate = self.curdate, date

            if hold:
                # 在有持仓的情况下，对净值每日更新计算
                self.update_port_netvalue(date)
        
        # 回测后进行的处理
        self.after_backtest()

    def update_port_netvalue(self, date):
        """
        更新每日净值
        根据权重和行情计算组合百分百变动
        """
        # 权重均为0被删掉了
        if date not in self.stock_weights.columns:
            self.portfolio_record[date] = 0.0
        else:
            weights = self.stock_weights.loc[:, date]
            weights = weights.dropna()
            weights /= np.sum(weights)
            codes = weights.index
            pct_chg = self.market_data.loc[codes, date].values
            cur_wt_pctchg = np.nansum(pct_chg * weights.values)
            self.portfolio_record[date] = cur_wt_pctchg

    def after_backtest(self):
        # 主要针对净值记录格式进行调整，将pctchg转换为净值数值；
        # 同时将持仓记录转化为矩
        self.portfolio_record = pd.DataFrame(self.portfolio_record, index=[0]).T

        basic = r'D:\pythoncode\IndexEnhancement'
        save_path = os.path.join(basic, '回测结果')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        hased = os.listdir(save_path)

        self.portfolio_record.columns = ['netval_pctchg']
        self.portfolio_record['net_value'] = (1 + self.portfolio_record['netval_pctchg']).cumprod()
        self.portfolio_record['benchmark'] = self._get_benchmark()
        self.portfolio_record['bh:net_value'] = (1 + self.portfolio_record['benchmark']).cumprod()
        self.portfolio_record['excess_ret'] = self.portfolio_record['net_value'] / self.portfolio_record['bh:net_value']

        plt.subplot(2, 1, 1)
        p1, = plt.plot(self.portfolio_record['net_value'], color='blue')
        p2, = plt.plot(self.portfolio_record['bh:net_value'], color='pink')
        plt.legend([p1, p2], ["策略净值", "基准净值"], loc='upper left')
        plt.subplot(2, 1, 2)
        ex, = plt.plot(self.portfolio_record['excess_ret'], color='red')
        plt.legend([ex], ["超额收益"], loc='upper left')
        save_name = '净值走势图_' + str(len(hased)//3)
        plt.savefig(os.path.join(save_path, save_name+'.png'))
        plt.show()

        save_name = '净值记录_' + str(len(hased)//3)
        self.portfolio_record.to_csv(os.path.join(save_path, save_name+'.csv'), encoding='gbk')

        # 分年度统计
        res_yearly = self.summary_yearly()

        save_name = '评价指标_' + str(len(hased)//3)
        res_yearly.to_csv(os.path.join(save_path, save_name+'.csv'), encoding='gbk')

    def summary(self, start_date=None, end_date=None):
        if start_date is None and end_date is None:
            start_date, end_date = self.portfolio_record.index[0], self.portfolio_record.index[-1]

        ann_ret = self._annual_return(start_date, end_date, None)      # 年化收益
        ann_vol = self._annual_vol(start_date, end_date, None)         # 年化波动
        max_wd = self._max_drawdown(start_date, end_date, None)        # 最大回撤
        sharpe = self._sharpe_ratio(start_date, end_date, ann_ret=ann_ret, ann_vol=ann_vol)              # 夏普比率
        ann_excess_ret = self._ann_excess_ret(start_date, end_date)    # 年化超额收益
        # ic_rate = self._ic_rate(start_date, end_date)
        win_rate = self._winning_rate(start_date, end_date)            # 相对基准日胜率
        # turnover_rate = self._turnover_rate(start_date, end_date)      # 换手率
        summary = {
                '年度收益': ann_ret, 
                '年度波动': ann_vol, 
                '最大回撤': max_wd, 
                '夏普比率': sharpe, 
                '年度超额收益': ann_excess_ret, 
                '胜率': win_rate,
                # '跟踪误差': te,
                # '换手率': turnover_rate,
                # '信息比率': ic_rate
                  }
        return pd.Series(summary)

    def summary_yearly(self):
        if len(self.portfolio_record) == 0:
            raise RuntimeError("请运行回测函数后再查看回测统计.")
        if 'benchmark' not in self.portfolio_record.columns:
            self.portfolio_record['benchmark'] = self._get_benchmark()
        
        all_dates = self.portfolio_record.index
        start_dates = all_dates[:1].tolist() + list(before_date for before_date, after_date in zip(all_dates[1:], all_dates[:-1])
                                          if before_date.year != after_date.year)
        end_dates = list(before_date for before_date, after_date in zip(all_dates[:-1], all_dates[1:])
                         if before_date.year != after_date.year) + all_dates[-1:].tolist()
        res = pd.DataFrame()

        for sdate, edate in zip(start_dates, end_dates):
            summary_year = self.summary(sdate, edate)
            summary_year.name = str(sdate.year)
            res = pd.concat([res, summary_year], axis=1)

        summary_all = self.summary()
        summary_all.name = '总计'
        res = pd.concat([res, summary_all], axis=1)
        res = res.T[['年度收益', '年度波动', '夏普比率', '年度超额收益',
                     '最大回撤', '胜率']]
        return res
    
    def _get_benchmark(self):
        start_date, end_date = self.portfolio_record.index[0], \
                                self.portfolio_record.index[-1]
        return self.benchmark_data.loc[start_date:end_date]
    
    def _get_date_gap(self, start_date=None, end_date=None, freq='d'):
        if start_date is None and end_date is None:
            start_date = self.portfolio_record.index[0]
            end_date = self.portfolio_record.index[-1]
        days = (end_date - start_date) / toffsets.timedelta(1)
        if freq == 'y':
            return days / 365
        elif freq == 'q':
            return days / 365 * 4
        elif freq == 'M':
            return days / 365 * 12
        elif freq == 'd':
            return days 
    
    def _te(self, start_date=None, end_date=None):
        if start_date and end_date:
            pr = self.portfolio_record.loc[start_date:end_date]
        else:
            pr = self.portfolio_record
        td = (pr['netval_pctchg'] - pr['benchmark'])
        te = np.sqrt(min(len(pr), 252)) * np.sqrt(1 / (len(td) - 1) * np.sum((td - np.mean(td))**2))
        return te
    
    def _ic_rate(self, start_date=None, end_date=None):
        excess_acc_ret = self._get_excess_acc_ret(start_date, end_date)
        ann_excess_ret = self._ann_excess_ret(start_date, end_date)
        ann_excess_ret_vol = self._annual_vol(excess_acc_ret, 
                                              start_date, end_date)
        return (ann_excess_ret - self.rf_rate) / ann_excess_ret_vol
    
    def _turnover_rate(self, start_date=None, end_date=None):            
        positions = self.position_record.fillna(0).T
        if start_date and end_date:
            positions = positions.loc[start_date:end_date]
        turnover_rate = np.sum(np.abs(positions - positions.shift(1)) / 2, axis=1)
        turnover_rate = np.mean(turnover_rate) * 12
        return turnover_rate
        
    def _winning_rate(self, start_date=None, end_date=None):

        nv_pctchg = self.portfolio_record['netval_pctchg']
        bm_pctchg = self.portfolio_record['benchmark']

        if start_date and end_date:
            nv_pctchg = nv_pctchg.loc[start_date:end_date]
            bm_pctchg = bm_pctchg.loc[start_date:end_date]

        win_daily = (nv_pctchg > bm_pctchg)
        win_rate = np.sum(win_daily) / len(win_daily)
        return win_rate
    
    def _annual_return(self, start_date, end_date, net_vals=None):
        if net_vals is None:           # 对月度频率的净值走势，应该使用前一个月末的数值
            net_vals = self.portfolio_record['net_value']

        tmp = list(net_vals.index)
        start_date_id = tmp.index(start_date)
        end_date_id = tmp.index(end_date)
        if start_date_id == 0:
            net_vals = net_vals[start_date_id:end_date_id+1]
            # 净值从1开始
            total_ret = net_vals.values[-1]/1 - 1
        else:
            # 对于月频数据，计算收益从start_date_id前一个
            net_vals = net_vals[start_date_id-1:end_date_id+1]
            total_ret = net_vals.values[-1]/net_vals.values[0] - 1

        date_gap = self._get_date_gap(start_date, end_date, freq='d')
        exp = 365 / date_gap
        ann_ret = (1 + total_ret) ** exp - 1
        if date_gap <= 365:
            return total_ret
        else:
            return ann_ret
    
    def _annual_vol(self, start_date, end_date, net_vals=None):
        if net_vals is None:
            ret_per_period = self.portfolio_record['netval_pctchg']

        ret_per_period = ret_per_period.loc[start_date:end_date]
        # 年化波动率 = 日频收益率标准差 * sqrt(250)
        # 年化波动率 = 周频收益率标准差 * sqrt(52)
        # 年化波动率 = 月频收益率标准差 * sqrt(12)
        ann_vol = ret_per_period.std() * np.sqrt(12)
        return ann_vol

    def _max_drawdown(self, start_date, end_date, acc_rets=None):
        if acc_rets is None:
            acc_rets = self.portfolio_record['net_value'] - 1
        acc_rets = acc_rets.loc[start_date:end_date]
        max_drawdown = (1 - (1 + acc_rets) / (1 + acc_rets.expanding().max()) ).max()
        return max_drawdown

    def _sharpe_ratio(self, start_date, end_date, ann_ret=None, ann_vol=None):
        if ann_ret is None:
            ann_ret = self._annual_return(start_date, end_date)
        if ann_vol is None:
            ann_vol = self._annual_vol(start_date, end_date)
        return (ann_ret - self.rf_rate) / ann_vol
    
    def _get_excess_acc_ret(self, start_date=None, end_date=None):

        bm_ret = self.portfolio_record['benchmark']
        nv_ret = self.portfolio_record['netval_pctchg']

        if start_date and end_date:
            bm_ret = bm_ret.loc[start_date:end_date] 
            nv_ret = nv_ret.loc[start_date:end_date]
        excess_ret = nv_ret.values.flatten() - bm_ret.values.flatten()
        excess_acc_ret = pd.Series(np.cumprod(1+excess_ret), index=nv_ret.index)
        return excess_acc_ret
        
    def _ann_excess_ret(self, start_date=None, end_date=None):
        excess_acc_ret = self._get_excess_acc_ret(start_date, end_date)
        ann_excess_ret = self._annual_return(start_date=start_date,
                                             end_date=end_date,
                                             net_vals=excess_acc_ret
                                             )
        return ann_excess_ret     


# 因子分层回测
class SingleFactorLayerDivisionBacktest:
    def __init__(self, *, factor_name, factor_data, num_layers=5, 
                 if_concise=True, pct_chg_nm, **kwargs):
        self.num_layers = num_layers              # 分层回测层数
        self.factor_name = factor_name            # 因子名称
        self.factor_data = factor_data            # 月频因子矩阵数据（行为证券代码，列为日期）
        self.stock_pool = self.factor_data.index  # 股票池
        self.if_concise = if_concise              # 是否使用简便回测方式，如是，则使用月涨跌幅进行回测，
                                                  # 否则采用日度复权价格进行回测
        self.pctchg_nm = pct_chg_nm
        self.kwargs = kwargs
            
    def run_layer_division_bt(self, equal_weight=True):
        # 运行分层回测
        if self.if_concise:
            result = self._run_rapid_layer_divbt()
        else:
            stock_weights = self.get_stock_weight(equal_weight)            # 获取各层权重
            result = pd.DataFrame()
            for i in range(self.num_layers):
                kwargs = deepcopy(self.kwargs)
                kwargs['stock_weights'] = stock_weights[i]
                bt = Backtest_stock(**kwargs)
                bt.run_backtest()
                bt.portfolio_record.index = [f'第{i+1}组']
                result = pd.concat([result, bt.portfolio_record.T], axis=1)
        # print(f"{self.factor_name}分层回测结束！")
            
        result.index.name = self.factor_name
        return result
    
    def _run_rapid_layer_divbt(self):
        result = pd.DataFrame()
        for date in self.pctchg_nm.columns:
            if date not in self.factor_data.columns:
                continue
            # 针对因子全部为nan的特殊情况
            if len(self.factor_data[date].dropna()) == 0:
                continue

            cur_weights = self.get_stock_weight_by_group(self.factor_data[date], True)
            cur_pctchg_nm = self.pctchg_nm[date] / 100
            group_monthly_ret = pd.Series()
            for group in cur_weights.columns:
                group_weights = cur_weights[group].dropna()
                cur_layer_stocks = group_weights.index
                group_monthly_ret.loc[group] = np.nanmean(cur_pctchg_nm.loc[cur_layer_stocks])
            group_monthly_ret.name = date
            result = pd.concat([result, group_monthly_ret], axis=1)

        if len(result.columns) < 5:
            return pd.DataFrame()

        # 对齐实际日期与对应月收益
        months = result.columns[1:].tolist()
        del result[months[-1]]
        result.columns = months
        return result.T
    
    def get_stock_weight(self, equal_weight=True):
        # 对权重的格式进行转换，以便后续回测
        dates = self.factor_data.columns
        stk_weights = [self.get_stock_weight_by_group(self.factor_data[date], 
                       equal_weight) for date in dates]
        result = {date: stk_weight for date, stk_weight in zip(dates, stk_weights)}
        result = pd.Panel.from_dict(result)
        result = [result.minor_xs(group) for group in result.minor_axis]
        return result
        
    def get_stock_weight_by_group(self, factor, equal_weight=False):
        # 根据因子的大小降序排列
        factor = factor.sort_values(ascending=False).dropna()
        # 计算获得各层权重

        weights = self.cal_weight(factor.index)
        result = pd.DataFrame(index=factor.index)
        result.index.name = 'code'
        for i in range(len(weights)):
            labels = [factor.index[num] for num, weight in weights[i]]
            values = [weight for num, weight in weights[i]]
            result.loc[labels, f'第{i+1}组'] = values
        if equal_weight:
            # 设置为等权
            result = result.where(pd.isnull(result), 1)
        return result
            
    def cal_weight(self, stock_pool):
        # 权重计算方法参考华泰证券多因子系列研报
        total_num = len(stock_pool)
        weights = []
        
        total_weights = 0; j = 0
        for i in range(total_num):
            total_weights += 1 / total_num
            if i == 0:
                weights.append([])
            if total_weights > len(weights) * 1 / self.num_layers:
                before = i, len(weights) * 1 / self.num_layers - \
                            sum(n for k in range(j+1) for m, n in weights[k]) 
                after = i, 1 / total_num - before[1]
                
                weights[j].append(before) 
                weights.append([])
                weights[j+1].append(after)
                j += 1
            else:
                cur = i, 1 / total_num
                weights[j].append(cur)
        
        # 调整尾差
        if len(weights[-1]) == 1:
            weights.remove(weights[-1])

        return weights


def panel_to_matrix(factors, factor_path, save_path):
    """
    将经过预处理的因子截面数据转换为因子矩阵数据
    """
    global industry_benchmark
    factors_to_be_saved = [f.replace('/','_div_') for f in factors]
    factor_matrix_path = os.path.join(save_path, '因子矩阵') if not save_path.endswith('因子矩阵') else save_path 
    if not os.path.exists(factor_matrix_path):
        os.mkdir(factor_matrix_path)
    # 下面的条件退出的代码就不是可以容纳更新的逻辑
    # else:
    #     factors = set(tuple(factors_to_be_saved)) - \
    #         set(f.split('.')[0] for f in os.listdir(factor_matrix_path))
    #     if len(factors) == 0:
    #         return None
        
    factors = sorted(f.replace('_div_', '/') for f in factors) 
    if '预处理' in factor_path:
        factors.extend(['Pct_chg_nm', f'Industry_sw',
                        'Mkt_cap_float',  'Second_industry'])
    datpanel = {}
    for f in os.listdir(factor_path):
        open_name = f.replace('_div_','/')
        datdf = pd.read_csv(os.path.join(factor_path, open_name), encoding='gbk', engine='python')

        if 'Code' in datdf.columns:
            datdf = datdf.set_index('Code')
        elif 'code' in datdf.columns:
            datdf = datdf.set_index('code')
        elif 'Name' in datdf.columns:
            datdf = datdf.set_index('Name')
        else:
            print('index error')

        date = pd.to_datetime(f.split('.')[0])
        factors_tmp = [fa for fa in factors if fa in datdf.columns]
        datpanel[date] = datdf[factors_tmp]

    datpanel = pd.Panel(datpanel)
    datpanel = datpanel.swapaxes(0, 2)
    for factor in datpanel.items:
        dat = datpanel.loc[factor]
        save_name = factor.replace('/', '_div_') if '/' in factor else factor
        dat.to_csv(os.path.join(factor_matrix_path, save_name+'.csv'),
                   encoding='gbk')


def plot_layerdivision(pathdict, records, fname, concise):

    save_path = pathdict['sf_test_save_path']
    layerdiv_figpath = os.path.join(save_path, '分层回测', '分层图')

    if not os.path.exists(layerdiv_figpath):
        os.makedirs(layerdiv_figpath)
    
    if concise:
        records = np.cumprod(1+records)
        records /= records.iloc[0]
    records = records.T / records.apply(np.mean, axis=1)
    records = records.T
    plt.plot(records)
    plt.title(fname)
    plt.legend(records.columns, loc=0)
    
    save_name = fname.replace('/', '_div_') if '/' in fname else fname

    plt.savefig(os.path.join(layerdiv_figpath, save_name+f'_{len(records.columns)}.png'))
    plt.close()


def bar_plot_yearly(pathdict, records, fname, concise):

    barwidth = 1 / len(records.columns) - 0.03

    save_path = pathdict['sf_test_save_path']
    layerdiv_barpath = os.path.join(save_path, '分层回测', '分年收益图')
    if not os.path.exists(layerdiv_barpath):
        os.mkdir(layerdiv_barpath)
    
    if concise:
        records_gp = records.groupby(pd.Grouper(freq='y'))
        records = pd.DataFrame()
        for year, month_ret in records_gp:
            month_netvalue = np.cumprod(1+month_ret)
            try:
                year_return = month_netvalue.iloc[-1] / month_netvalue.iloc[0] - 1
            except Exception as e:
                pass
            year_return.name = year
            if year == 2009:
                year_return = (1 + year_return) ** (12/11) - 1
            records = pd.concat([records, year_return], axis=1)
        records = records.T
    else:
        records = records.groupby(pd.Grouper(freq='y')).\
                apply(lambda df: df.iloc[-1] / df.iloc[0] - 1)
        records = records.T - records.mean(axis=1)
        records = records.T
    # 减去5组间均值
    records = records.T - records.mean(axis=1)
    records = records.T
    time = np.array([d.year for d in records.index])

    plt.bar(time, records['第1组'], barwidth, color='blue', label='第1组')
    plt.bar(time+barwidth, records['第2组'], barwidth, color='green', label='第2组')
    if len(records.columns) > 2:
        plt.bar(time+2*barwidth, records['第3组'], barwidth, color='red', label='第3组')
    if len(records.columns) > 3:
        plt.bar(time+3*barwidth, records['第4组'], barwidth, color='#E066FF', label='第4组')
    if len(records.columns) > 4:
        plt.bar(time+4*barwidth, records['第5组'], barwidth, color='#EEB422', label='第5组')

    plt.xticks(time+2.5*barwidth, time)
    plt.legend(records.columns, loc=0)

    save_name = fname.replace('/','_div_') if '/' in fname else fname
    plt.savefig(os.path.join(layerdiv_barpath, save_name+f'_{len(records.columns)}.png'))
    plt.close()


def plot_group_diff_plot(pathdict, records, fname, concise):
    global tick_spacing1

    save_path = pathdict['sf_test_save_path']
    layerdiv_diffpath = os.path.join(save_path, '分层回测', '组1-组5')
    if not os.path.exists(layerdiv_diffpath):
        os.mkdir(layerdiv_diffpath)

    num_layers = len(records.columns)
    if concise:
        records = np.cumprod(1+records)
        records /= records.iloc[0]

    tmp = '第' + str(num_layers) + '组'
    records = (records['第1组'] - records[tmp]) / records['第1组']
    
    time = [str(d)[:10] for d in records.index]
    
    fig, ax = plt.subplots(1,1)
    ax.plot(time, records.values)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing1))
    ax.set_title(fname)

    save_name = fname.replace('/','_div_') if '/' in fname else fname
    fig.savefig(os.path.join(layerdiv_diffpath, save_name+f'_{num_layers}.png'))
    plt.close()
    

