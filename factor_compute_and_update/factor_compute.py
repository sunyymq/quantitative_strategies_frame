import copy
import sys
import pandas as pd
import numpy as np
import os
from itertools import chain
from functools import reduce
import statsmodels.api as sm
import shelve
from datetime import datetime, timedelta
# from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from utility.tool0 import Data, scaler
from utility.factor_data_preprocess import add_to_panels, align
from utility.relate_to_tushare import generate_months_ends, trade_days
from utility.tool1 import CALFUNC, _calculate_su_simple, parallelcal,  lazyproperty, time_decorator, \
    get_signal_season_value, get_fill_vals, linear_interpolate, get_season_mean_value
from utility.constant import data_dair, root_dair
from utility.tool3 import adjust_months, add_to_panels, append_df


START_YEAR = 2009


class Factor_Compute(CALFUNC):

    def __init__(self, status):
        super().__init__()
        # status = 'update' 表示仅对已有的因子值进行更新， ='all' 表示全部重新计算
        self._mes = generate_months_ends()
        self._status = status

    def _get_update_month(self, fn):
        factor_m = eval('self.' + fn)
        # factor_m = self.RETURN_12M
        last_dt = pd.to_datetime(factor_m.columns[-1])
        to_update_month_list = [i for i in self._mes if i > last_dt]
        if len(to_update_month_list) == 0:
            print('没有更新必要')
            return None
            # sys.exit()
        else:
            return to_update_month_list

    @lazyproperty
    def compute_pct_chg_nm(self):

        pct = self.changepct_daily
        pct = 1 + pct/100

        mes1 = [m for m in self._mes if m in pct.columns]
        pct_chg = pd.DataFrame()
        for m in mes1:
            cols = [c for c in pct.columns if c.year == m.year and c.month == m.month]
            tmp_df = pct[cols]
            tmp_cum = tmp_df.cumprod(axis=1)
            res_df_t = tmp_cum[[tmp_cum.columns[-1]]] - 1
            pct_chg = pd.concat([pct_chg, res_df_t], axis=1)

        pct_chg = pct_chg * 100
        pct_chg_nm = pct_chg.shift(-1, axis=1)
        pct_chg_nm = CALFUNC.del_dat_early_than(pct_chg_nm, START_YEAR)

        return pct_chg_nm

    @lazyproperty
    def compute_pct_chg_nw(self):

        pct = self.changepct_daily
        pct = 1 + pct / 100

        tds = trade_days(freq='w')

        wes = [d for d in tds if d in pct.columns]
        pct_chg = pd.DataFrame()
        for i in range(1, len(wes)):
            # wes[i-1]   # 上周末的日期
            # wes[i]     # 本周末的日期
            last_we = np.where(self.changepct_daily.columns == wes[i-1])[0][0]
            we = np.where(self.changepct_daily.columns == wes[i])[0][0]
            week_range = self.changepct_daily.columns[last_we+1: we+1]
            tmp_df = pct[week_range]
            tmp_cum = tmp_df.cumprod(axis=1)
            res_df_t = tmp_cum[[tmp_cum.columns[-1]]] - 1
            pct_chg = pd.concat([pct_chg, res_df_t], axis=1)

        pct_chg = pct_chg * 100
        pct_chg_nw = pct_chg.shift(-1, axis=1)
        pct_chg_nw = CALFUNC.del_dat_early_than(pct_chg_nw, START_YEAR)

        return pct_chg_nw

    @lazyproperty
    def is_open1(self):
        open = self.openprice_daily
        high = self.highprice_daily
        low = self.lowprice_daily

        # 不是停牌的
        is_open = ~pd.isna(open)
        # 不是开盘涨跌停的
        tmp1 = open == high
        tmp2 = high == low
        tmp = ~(tmp1 & tmp2)

        is_open = tmp & is_open

        return is_open

    @lazyproperty
    def liquidity_barra(self):

        totalmv = self.totalmv_daily                 # 流通市值（万元）
        turnovervalue = self.turnovervalue_daily     # 成交额（万元）

        totalmv, turnovervalue = align(totalmv, turnovervalue)

        share_turnover = turnovervalue / totalmv

        new_mes = [m for m in self._mes if m in share_turnover.columns]

        def t_fun(tmp_df, freq=1):
            tmp_ar = tmp_df.values
            sentinel = -1e10
            res = np.log(np.nansum(tmp_ar, axis=1) / freq)
            res = np.where(np.isinf(res), sentinel, res)
            res_df = pd.DataFrame(data=res, index=tmp_df.index, columns=[tmp_df.columns[-1]])
            return res_df

        stom = pd.DataFrame()
        stoq = pd.DataFrame()
        stoa = pd.DataFrame()

        for m in new_mes:
            loc = np.where(share_turnover.columns == m)[0][0]
            if loc > 12 * 21:
                res_df1 = t_fun(share_turnover.iloc[:, loc+1 - 21:loc+1], 1)
                res_df3 = t_fun(share_turnover.iloc[:, loc+1 - 3*21:loc+1], 3)
                res_df12 = t_fun(share_turnover.iloc[:, loc+1 - 12*21:loc+1], 12)

                stom = pd.concat([stom, res_df1], axis=1)
                stoq = pd.concat([stoq, res_df3], axis=1)
                stoa = pd.concat([stoa, res_df12], axis=1)

        stom = CALFUNC.del_dat_early_than(stom, START_YEAR)
        stoq = CALFUNC.del_dat_early_than(stoq, START_YEAR)
        stoa = CALFUNC.del_dat_early_than(stoa, START_YEAR)

        res_dict = {"STOM_BARRA": stom,
                    "STOQ_BARRA": stoq,
                    "STOA_BARRA": stoa,
                    }

        return res_dict

    @lazyproperty
    def West_netprofit_yoy(self):
        west = self.west_netprofit_yoy
        west.fillna(0, inplace=True)
        return west

    @lazyproperty
    def beta(self):

        y = self.changepct_daily.T / 100
        index_p = self.index_price_daily
        index_p = index_p.loc['HS300', :]
        index_r = index_p/index_p.shift(1) - 1
        index_r = index_r.dropna()

        new_index = [i for i in y.index if i in index_r.index]
        y = y.loc[new_index, :]

        new_mes = [m for m in self._mes if m in y.index and np.where(y.index == m)[0][0] > 504]
        b, alpha, sigma = self._rolling_regress(y, index_r, window=504, half_life=252, target_date=new_mes)

        return b

    # 流通市值
    @lazyproperty
    def Mkt_cap_float(self):
        negotiablemv = self.negotiablemv_daily
        negotiablemv = CALFUNC.d_freq_to_m_freq(negotiablemv)
        res = CALFUNC.del_dat_early_than(negotiablemv, START_YEAR)
        return res

    # 规模因子
    @lazyproperty
    def LNCAP_Barra(self):
        lncap = np.log(self.negotiablemv_daily * 10000)
        lncap = CALFUNC.d_freq_to_m_freq(lncap)
        lncap = CALFUNC.del_dat_early_than(lncap, START_YEAR)
        return lncap

    @lazyproperty
    def MIDCAP_barra(self):
        lncap = np.log(self.negotiablemv_daily * 10000)
        lncap = CALFUNC.d_freq_to_m_freq(lncap)
        y = lncap ** 3
        X = lncap
        y = y.T
        X = X.T

        resid = pd.DataFrame()
        for code in y.columns:
            y_ = y[[code]]
            x_ = X[[code]]
            x_['const'] = 1
            dat = pd.concat([x_, y_], axis=1)
            dat = dat.dropna(how='any', axis=0)
            X_, y_ = dat.iloc[:, :-1], dat.iloc[:, -1:]

            if len(y_) > 0:
                model = sm.WLS(y_, X_)
                result = model.fit()

                params_ = result.params
                resid_ = y_ - pd.DataFrame(np.dot(X_, params_), index=y_.index,
                                           columns=[code])
            else:
                resid_ = pd.DataFrame([np.nan] * len(y), index=y.index, columns=[code])

            resid = pd.concat([resid, resid_], axis=1)

        resid = resid.T
        resid = CALFUNC.del_dat_early_than(resid, START_YEAR)

        return resid

    @lazyproperty
    def Indmom(self):
        window = 6 * 21
        half_life = 21
        cd = self.changepct_daily
        logret = np.log(1 + cd.fillna(0) / 100)

        weight = self._get_exp_weight(window, half_life)

        def func0(se, wei):
            return np.dot(se.values, wei)

        rs = pd.DataFrame()
        for m in self._mes:
            if np.any(logret.columns == m):
                loc = np.where(logret.columns == m)[0][0]
                if loc > window:
                    change_in_window = logret.iloc[:, loc + 1 - window:loc + 1]
                    tmp = change_in_window.apply(func0, args=(weight,), axis=1)
                    tmp_se = pd.DataFrame({m: tmp})
                    rs = pd.concat([rs, tmp_se], axis=1)

        basic_inform = self.stock_basic_inform
        first_ind = basic_inform[['申万一级行业']]
        cap_sqrt = np.sqrt(self.negotiablemv_daily)

        rs, cap_sqrt = align(rs, cap_sqrt)

        grouped = first_ind.groupby('申万一级行业')

        indmom = pd.DataFrame()
        for ind, v in grouped:
            codes = list(v.index)
            rs_tmp = rs.loc[codes, :]
            cap_tmp = cap_sqrt.loc[codes, :]

            ind_rs = rs_tmp * cap_tmp
            ind_sun = ind_rs.sum()

            for i, se in ind_rs.iterrows():
                imdmom_tmp = -1 * (se - ind_sun)
                imdmom_tmp1 = pd.DataFrame({i: imdmom_tmp}).T
                indmom = pd.concat([indmom, imdmom_tmp1], axis=0)

        indmom = indmom.sort_index()
        return indmom

    @lazyproperty
    def Std_nm(self):
        # n分别为1、3、6、12，每个月为21个交易日
        pct = self.changepct_daily/100

        new_mes = [m for m in self._mes if m in pct.columns]

        std_1m = pd.DataFrame()
        std_3m = pd.DataFrame()
        std_6m = pd.DataFrame()
        std_12m = pd.DataFrame()

        for m in new_mes:
            loc = np.where(pct.columns == m)[0][0]
            if loc > 12 * 21:
                res_df1 = pct.iloc[:, loc + 1 - 21:loc + 1].std(axis=1)        # 对DF使用std，会自动处理nan
                res_df3 = pct.iloc[:, loc + 1 - 3 * 21:loc + 1].std(axis=1)
                res_df6 = pct.iloc[:, loc + 1 - 3 * 21:loc + 1].std(axis=1)
                res_df12 = pct.iloc[:, loc + 1 - 12 * 21:loc + 1].std(axis=1)

                std_1m = pd.concat([std_1m, pd.DataFrame({m: res_df1})], axis=1)
                std_3m = pd.concat([std_3m, pd.DataFrame({m: res_df3})], axis=1)
                std_6m = pd.concat([std_6m, pd.DataFrame({m: res_df6})], axis=1)
                std_12m = pd.concat([std_12m, pd.DataFrame({m: res_df12})], axis=1)

        std_1m = CALFUNC.del_dat_early_than(std_1m, START_YEAR)
        std_3m = CALFUNC.del_dat_early_than(std_3m, START_YEAR)
        std_6m = CALFUNC.del_dat_early_than(std_6m, START_YEAR)
        std_12m = CALFUNC.del_dat_early_than(std_12m, START_YEAR)

        res_dict = {"Std_1m": std_1m,
                    "Std_3m": std_3m,
                    "Std_6m": std_6m,
                    "Std_12m": std_12m,
                    }

        return res_dict

    # 盈利预测机构家数
    @lazyproperty
    def Est_instnum(self):
        est_instnum = self.est_instnum
        # 该数据缺失较多，对Nan的直接用0替代
        est_instnum = est_instnum.fillna(0)
        # 取根号
        est_instnum = est_instnum ** 0.5
        est_instnum = append_df(est_instnum)
        return est_instnum

    @lazyproperty
    # # 估值因子
    def ep(self):
        pe_daily = self.pe_daily
        pe = CALFUNC.d_freq_to_m_freq(pe_daily)
        ep = 1/pe
        res = CALFUNC.del_dat_early_than(ep, START_YEAR)

        return res

    @lazyproperty
    def bp(self):
        pb_daily = self.pb_daily
        pb = CALFUNC.d_freq_to_m_freq(pb_daily)
        bp = 1 / pb
        res = CALFUNC.del_dat_early_than(bp, START_YEAR)

        return res

    @lazyproperty
    def assetturnover_q(self):
        totalassets = self.totalassets
        revenue = self.operatingrevenue
        # 得到单季度 净利润
        sig_season_revenue = get_signal_season_value(revenue)
        # 得到季度平均总资产
        s_mean_totalassets = get_season_mean_value(totalassets)

        turnover_q = (sig_season_revenue / s_mean_totalassets) * 100
        turnover_q = adjust_months(turnover_q)
        turnover_q = append_df(turnover_q)
        turnover_q = CALFUNC.del_dat_early_than(turnover_q, START_YEAR)

        return turnover_q

    @lazyproperty
    def totalassetturnover(self):

        totalassettrate = self.totalassettrate
        tmp0 = adjust_months(totalassettrate)
        tmp1 = append_df(tmp0)
        res = CALFUNC.del_dat_early_than(tmp1, START_YEAR)

        return res

    @lazyproperty
    # 单季度毛利率
    def grossprofitmargin_q(self):
        '''
        计算公示：（营业收入 - 营业成本） / 营业收入 * 100 %
        计算单季度指标，应该先对 营业收入 和 营业成本 分别计算单季度指标，再计算
        '''
        revenue = self.operatingrevenue    # 营业收入
        cost = self.operatingcost       # 营业成本
        # 财务指标常规处理，移动月份，改月末日期
        revenue_q = get_signal_season_value(revenue)
        cost_q = get_signal_season_value(cost)
        gross_q = (revenue_q - cost_q) / revenue_q
        # 调整为公告日期
        tmp = adjust_months(gross_q)
        # 用来扩展月度数据
        tmp = append_df(tmp)
        res = CALFUNC.del_dat_early_than(tmp, START_YEAR)
        return res

    @lazyproperty
    # 毛利率ttm
    def grossprofitmargin_ttm(self):

        gir = self.grossincomeratiottm
        gir = adjust_months(gir)
        # 用来扩展月度数据
        gir = append_df(gir)
        res = CALFUNC.del_dat_early_than(gir, START_YEAR)
        return res

    @lazyproperty
    def peg(self):
        # PEG = PE / 过去12个月的EPS增长率
        pe_daily = self.pe_daily
        basicepsyoy = self.basicepsyoy
        basicepsyoy = adjust_months(basicepsyoy)
        epsyoy = append_df(basicepsyoy, target_feq='D', fill_type='preceding')

        pe_daily = CALFUNC.del_dat_early_than(pe_daily, START_YEAR)
        epsyoy = CALFUNC.del_dat_early_than(epsyoy, START_YEAR)

        [pe_daily, epsyoy] = align(pe_daily, epsyoy)

        [h, l] = pe_daily.shape
        pe_ar = pe_daily.values
        eps_ar = epsyoy.values

        res = np.zeros([h, l])
        for i in range(0, h):
            for j in range(0, l):
                if pd.isna(eps_ar[i, j]) or eps_ar[i, j] == 0:
                    res[i, j] = np.nan
                else:
                    res[i, j] = pe_ar[i, j] / eps_ar[i, j]

        res_df = pd.DataFrame(data=res, index=pe_daily.index, columns=pe_daily.columns)

        return res_df

    @lazyproperty
    # 毛利率季度改善
    def grossprofitmargin_diff(self):
        revenue = self.operatingrevenue  # 营业收入
        cost = self.operatingcost  # 营业成本
        # 财务指标常规处理，移动月份，改月末日期
        revenue_q = get_signal_season_value(revenue)
        cost_q = get_signal_season_value(cost)
        gross_q = (revenue_q - cost_q) / revenue_q

        gir_d = CALFUNC.generate_diff(gross_q)
        gir_d = adjust_months(gir_d)
        # 用来扩展月度数据
        gir_d = append_df(gir_d)
        res = CALFUNC.del_dat_early_than(gir_d, START_YEAR)
        return res

    # Mom
    @lazyproperty
    def return_n_m(self):
        close = self.closeprice_daily
        adj = self.adjfactor

        close, adj = align(close, adj)
        c_p = close*adj
        c_v = c_p.values
        hh, ll = c_v.shape

        # 1个月、3个月、6个月、12个月
        m1 = np.zeros(c_v.shape)
        m3 = np.zeros(c_v.shape)
        m6 = np.zeros(c_v.shape)
        m12 = np.zeros(c_v.shape)
        for i in range(21, ll):
            m1[:, i] = c_v[:, i]/c_v[:, i-21]
        for i in range(21*3, ll):
            m3[:, i] = c_v[:, i]/c_v[:, i-21*3]
        for i in range(21*6, ll):
            m6[:, i] = c_v[:, i]/c_v[:, i-21*6]
        for i in range(21*12, ll):
            m12[:, i] = c_v[:, i]/c_v[:, i-21*12]

        m1_df = pd.DataFrame(data=m1, index=c_p.index, columns=c_p.columns)
        m3_df = pd.DataFrame(data=m3, index=c_p.index, columns=c_p.columns)
        m6_df = pd.DataFrame(data=m6, index=c_p.index, columns=c_p.columns)
        m12_df = pd.DataFrame(data=m12, index=c_p.index, columns=c_p.columns)

        m1_df_m = CALFUNC.d_freq_to_m_freq(m1_df)
        m3_df_m = CALFUNC.d_freq_to_m_freq(m3_df)
        m6_df_m = CALFUNC.d_freq_to_m_freq(m6_df)
        m12_df_m = CALFUNC.d_freq_to_m_freq(m12_df)

        m1_df_m1 = CALFUNC.del_dat_early_than(m1_df_m, START_YEAR)
        m3_df_m1 = CALFUNC.del_dat_early_than(m3_df_m, START_YEAR)
        m6_df_m1 = CALFUNC.del_dat_early_than(m6_df_m, START_YEAR)
        m12_df_m1 = CALFUNC.del_dat_early_than(m12_df_m, START_YEAR)

        res_dict = {'RETURN_1M': m1_df_m1 - 1,
                    'RETURN_3M': m3_df_m1 - 1,
                    'RETURN_6M': m6_df_m1 - 1,
                    'RETURN_12M': m12_df_m1 - 1,
                    }

        return res_dict

    @lazyproperty
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

    @lazyproperty
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

    # 盈利
    @lazyproperty
    def ROA_ttm(self):
        roa_ttm = self.roattm
        roa_ttm = adjust_months(roa_ttm)
        roa_ttm = append_df(roa_ttm)
        roa_ttm = CALFUNC.del_dat_early_than(roa_ttm, START_YEAR)
        return roa_ttm

    # todo
    @lazyproperty
    def ROA_q(self):
        totalassets = self.totalassets
        netprofit = self.netprofit
        # 得到单季度 净利润
        sig_season_netprofit = get_signal_season_value(netprofit)
        # 得到季度平均总资产
        s_mean_totalassets = get_season_mean_value(totalassets)

        roa_q = (sig_season_netprofit/s_mean_totalassets) * 100
        roa_q = adjust_months(roa_q)
        roa_q = append_df(roa_q)
        roa_q = CALFUNC.del_dat_early_than(roa_q, START_YEAR)
        return roa_q

    @lazyproperty
    def ROE_q(self):
        totalshareholderequity = self.totalshareholderequity
        netprofit = self.netprofit
        # 得到单季度 净利润
        sig_season_netprofit = get_signal_season_value(netprofit)
        # 得到季度平均总资产
        s_mean_equity = get_season_mean_value(totalshareholderequity)

        roe_q = (sig_season_netprofit / s_mean_equity) * 100
        roe_q = adjust_months(roe_q)
        roe_q = append_df(roe_q)
        roe_q = CALFUNC.del_dat_early_than(roe_q, START_YEAR)
        return roe_q

    @lazyproperty
    def Profitmargin_q(self):     # 单季度净利润率
        '''
        1.qfa_deductedprofit：单季度.扣除非经常损益后的净利润
        2.qfa_oper_rev： 单季度.营业收入
        :return:
        '''

        netprofit = self.netprofitcut               # 扣除非经常损益后的净利润
        operatingrevenue = self.operatingrevenue
        sig_season_netprofit = get_signal_season_value(netprofit)
        sig_season_operatingrevenue = get_signal_season_value(operatingrevenue)
        profitmargin_q = sig_season_netprofit/sig_season_operatingrevenue
        profitmargin_q = adjust_months(profitmargin_q)
        profitmargin_q = append_df(profitmargin_q)

        pq = CALFUNC.del_dat_early_than(profitmargin_q, START_YEAR)

        return pq

    # 成长
    @lazyproperty
    def Profit_G_q(self):     # qfa_yoyprofit：单季度.净利润同比增长率
        netprofit = self.netprofitcut               # 扣除非经常损益后的净利润
        sig_season_netprofit = get_signal_season_value(netprofit)
        p_g = CALFUNC.generate_yoygr(sig_season_netprofit)
        p_g = adjust_months(p_g)
        p_g = append_df(p_g)
        profit_g_q = CALFUNC.del_dat_early_than(p_g, START_YEAR)
        return profit_g_q

    @lazyproperty
    def ROE_G_q(self):        # 单季度.ROE同比增长率
        roe = self.roe
        sig_season_roe = get_signal_season_value(roe)
        roe_g = CALFUNC.generate_yoygr(sig_season_roe)
        roe_g = adjust_months(roe_g)
        roe_g = append_df(roe_g)
        roe_g_q = CALFUNC.del_dat_early_than(roe_g, START_YEAR)
        return roe_g_q

    @lazyproperty
    def Sales_G_q(self):      # qfa_yoysales：单季度.营业收入同比增长率
        operatingrevenue = self.operatingrevenue
        sig_season_operatingrevenue = get_signal_season_value(operatingrevenue)
        sales_g = CALFUNC.generate_yoygr(sig_season_operatingrevenue)
        sales_g = adjust_months(sales_g)
        sales_g = append_df(sales_g)
        sales_g = CALFUNC.del_dat_early_than(sales_g, START_YEAR)
        return sales_g

    @lazyproperty
    def Rps(self):
        data = Data()

        all_codes = data.stock_basic_inform
        all_codes = pd.to_datetime(all_codes['ipo_date'.upper()])

        close_daily = data.closeprice_daily
        adjfactor = data.adjfactor
        close_price = close_daily*adjfactor
        close_price.dropna(axis=1, how='all', inplace=True)

        # 剔除上市一年以内的情况，把上市二年以内的股票数据都设为nan
        for i, row in close_price.iterrows():
            if i not in all_codes.index:
                row[:] = np.nan
                continue

            d = all_codes[i]
            row[row.index[row.index < d + timedelta(200)]] = np.nan

        if self._status == 'all':
            ext_120 = close_price/close_price.shift(periods=120, axis=1)
            ext_120.dropna(how='all', axis=1, inplace=True)
            rps_120 = ext_120.apply(scaler, scaler_max=100, scaler_min=1)

            rps = rps_120
            rps.dropna(how='all', axis=1, inplace=True)
            res = rps.apply(scaler, scaler_max=100, scaler_min=1)

            res = CALFUNC.del_dat_early_than(res, START_YEAR)
        elif self._status == 'update':
            hased_rps = data.RPS
            to_update = [col for col in close_price.columns if col not in hased_rps.columns and col > hased_rps.columns[-1]]
            if len(to_update) == 0:
                print('RPS无需要更新的部分')
                return hased_rps

            st = to_update[0]
            st_loc = np.where(close_price.columns == st)[0][0]
            st_loc = st_loc - 121

            close_price_new = close_price.iloc[:, st_loc:]
            ext_120 = close_price_new / close_price_new.shift(periods=120, axis=1)
            ext_120.dropna(how='all', axis=1, inplace=True)
            rps_120 = ext_120.apply(scaler, scaler_max=100, scaler_min=1)
            rps_120.dropna(how='all', axis=1, inplace=True)
            res0 = rps_120.apply(scaler, scaler_max=100, scaler_min=1)

            hased_rps[res0.columns] = res0
            res = hased_rps

        return res

    # 研发支出占营业收入的比例，因研发支出数据是在2018年3季度以后才开始披露的，所以该数据是在2018年3季度以后才有
    @lazyproperty
    def RDtosales(self):
        data = Data()

        rd_exp = data.rd_exp
        revenue = data.operatingrevenue
        rd_exp = CALFUNC.del_dat_early_than(rd_exp, 2018)
        revenue = CALFUNC.del_dat_early_than(revenue, 2018)

        res = rd_exp/revenue
        res = adjust_months(res)
        res = append_df(res)

        to_del = res.columns[res.isna().sum() / len(res) > 0.9]
        res.drop(to_del, axis=1, inplace=True)

        return res

    @lazyproperty
    def turn_nm(self):
        # n分别为1、3、6、12，每个月为21个交易日
        # turnover_ratio	换手率	decimal(10,4)		单位：％
        turnover = self.turnoverrate_daily / 100
        new_mes = [m for m in self._mes if m in turnover.columns]
        turn_1m = pd.DataFrame()
        turn_3m = pd.DataFrame()
        turn_6m = pd.DataFrame()
        turn_12m = pd.DataFrame()
        for m in new_mes:
            loc = np.where(turnover.columns == m)[0][0]
            if loc > 12 * 21:
                # t = turnover.iloc[:, loc + 1 - 21:loc + 1]
                res_df1 = turnover.iloc[:, loc + 1 - 21:loc + 1].mean(axis=1)
                res_df3 = turnover.iloc[:, loc + 1 - 3 * 21:loc + 1].mean(axis=1)
                res_df6 = turnover.iloc[:, loc + 1 - 6 * 21:loc + 1].mean(axis=1)
                res_df12 = turnover.iloc[:, loc + 1 - 12 * 21:loc + 1].mean(axis=1)

                turn_1m = pd.concat([turn_1m, pd.DataFrame({m: res_df1})], axis=1)
                turn_3m = pd.concat([turn_3m, pd.DataFrame({m: res_df3})], axis=1)
                turn_6m = pd.concat([turn_6m, pd.DataFrame({m: res_df6})], axis=1)
                turn_12m = pd.concat([turn_12m, pd.DataFrame({m: res_df12})], axis=1)

        turn_1m = CALFUNC.del_dat_early_than(turn_1m, START_YEAR)
        turn_3m = CALFUNC.del_dat_early_than(turn_3m, START_YEAR)
        turn_6m = CALFUNC.del_dat_early_than(turn_6m, START_YEAR)
        turn_12m = CALFUNC.del_dat_early_than(turn_12m, START_YEAR)

        res_dict = {"Turn_1m": turn_1m,
                    "Turn_3m": turn_3m,
                    "Turn_6m": turn_6m,
                    "Turn_12m": turn_12m,
                    }

        return res_dict

    @lazyproperty
    def Bias_turn_nm(self):
        # n分别为1、3、6、12，每个月为21个交易日
        # turnover_ratio	换手率	decimal(10,4)		单位：％
        turnover = self.turnoverrate_daily/100
        new_mes = [m for m in self._mes if m in turnover.columns]
        turn_1m = pd.DataFrame()
        turn_3m = pd.DataFrame()
        turn_6m = pd.DataFrame()
        turn_12m = pd.DataFrame()
        turn_24m = pd.DataFrame()
        for m in new_mes:
            loc = np.where(turnover.columns == m)[0][0]
            if loc > 12 * 21:
                res_df1 = turnover.iloc[:, loc + 1 - 21:loc + 1].mean(axis=1)
                res_df3 = turnover.iloc[:, loc + 1 - 3 * 21:loc + 1].mean(axis=1)
                res_df6 = turnover.iloc[:, loc + 1 - 6 * 21:loc + 1].mean(axis=1)
                res_df12 = turnover.iloc[:, loc + 1 - 12 * 21:loc + 1].mean(axis=1)
                res_df24 = turnover.iloc[:, loc + 1 - 24 * 21:loc + 1].mean(axis=1)
                turn_1m = pd.concat([turn_1m, pd.DataFrame({m: res_df1})], axis=1)
                turn_3m = pd.concat([turn_3m, pd.DataFrame({m: res_df3})], axis=1)
                turn_6m = pd.concat([turn_6m, pd.DataFrame({m: res_df6})], axis=1)
                turn_12m = pd.concat([turn_12m, pd.DataFrame({m: res_df12})], axis=1)
                turn_24m = pd.concat([turn_24m, pd.DataFrame({m: res_df24})], axis=1)

        turn_1m = CALFUNC.del_dat_early_than(turn_1m, START_YEAR)
        turn_3m = CALFUNC.del_dat_early_than(turn_3m, START_YEAR)
        turn_6m = CALFUNC.del_dat_early_than(turn_6m, START_YEAR)
        turn_12m = CALFUNC.del_dat_early_than(turn_12m, START_YEAR)
        turn_24m = CALFUNC.del_dat_early_than(turn_24m, START_YEAR)

        bias_turn_1m = turn_1m / turn_24m - 1
        bias_turn_3m = turn_3m / turn_24m - 1
        bias_turn_6m = turn_6m / turn_24m - 1
        bias_turn_12m = turn_12m / turn_24m - 1
        res_dict = {"Bias_turn_1m": bias_turn_1m,
                    "Bias_turn_3m": bias_turn_3m,
                    "Bias_turn_6m": bias_turn_6m,
                    "Bias_turn_12m": bias_turn_12m,
                    }
        return res_dict

    # 下面两个因子等权合成一个基金重仓股因子，应该在大消费、制造、TMT板块，周期类板块不适用，且因子处理时不做中性化处理。
    @lazyproperty
    def Fund_Topten(self):
        mes = generate_months_ends()
        data = Data()
        stock_basic_inform = data.stock_basic_inform

        # 读取基金复权净值数据
        adj_nav = data.fund_adj_nav

        # 读取流通市值数据
        negomv = data.negotiablemv_daily

        # 读取基金重仓股数据
        db = shelve.open(r'D:\Database_Stock\Data\fund\fund_portfolio.db')
        fund_dict = db['fund_portfolio_float_rate']
        db.close()

        # 基金复权净值与重仓股数据的基金取交集
        mult_funds = set(adj_nav.index) & set(fund_dict.keys())

        adj_nav = adj_nav.loc[mult_funds, :]

        to_del_keys = [k for k in fund_dict.keys() if k not in mult_funds]
        if len(to_del_keys) != 0:
            for k in to_del_keys:
                fund_dict.pop(k)

        # 根据净值数据，选择过去一年净值排名前50%的基金
        new_cols = [col for col in adj_nav.columns if col in mes and col > datetime(2007, 1, 1)]
        adj_nav = adj_nav[new_cols]
        adj_nav = adj_nav.dropna(how='all', axis=0)

        # 得到日期为key, 过去一年的区间收益率排名前50%基金代码list为value的dict
        top_50p_dict = {}
        for i in range(12, len(adj_nav.columns)):
            interval_rat = adj_nav[adj_nav.columns[i]] / adj_nav[adj_nav.columns[i - 12]]
            interval_rat = interval_rat.dropna()
            interval_rat = interval_rat.sort_values(ascending=False)

            top_50p_dict.update({adj_nav.columns[i]: list(interval_rat.index[:int(len(interval_rat) * 0.5)])})

        def date_compare(target, d_list):
            res = None
            for d in d_list:
                if target.year == d.year and target.month == d.month:
                    res = d
                    break
            return res

        # 根据这些基金的持仓，计算持有股票的市值比例
        ratio_df = pd.DataFrame(0, index=stock_basic_inform.index, columns=mes)
        for key, hold_df in fund_dict.items():
            for col in hold_df.columns:

                # 披露时间早于月末时间。
                # 太早的数据删除
                fid = date_compare(col, list(top_50p_dict.keys()))
                if not fid:
                    continue

                # 检查该期是否为业绩前50%
                if key not in top_50p_dict[fid]:
                    continue

                # 选择出对应的月份
                finded = 0
                for all_col in ratio_df.columns:
                    if all_col.year == col.year and all_col.month == col.month:
                        finded = all_col
                        break

                # 太早的数据，如02年的
                if finded == 0:
                    continue

                # 一个小bug
                if col.month not in [1, 4, 7, 10]:
                    continue

                target_month = ratio_df[finded]

                tmp_se = hold_df[col].dropna()
                # 针对对应的股票，市值相加
                for stock in tmp_se.index:
                    if stock in target_month.index:
                        target_month[stock] = target_month[stock] + tmp_se[stock]

                # target_month.sum()
                ratio_df[finded] = target_month

            test = (ratio_df > 0).sum().sum()

        test = (ratio_df > 0).sum()

        # 对于空余的月份，直接复制前值
        for col_n in range(1, len(ratio_df.columns)):
            if ratio_df[ratio_df.columns[col_n]].sum() == 0 and \
                    ratio_df[ratio_df.columns[col_n - 1]].sum() != 0:
                ratio_df[ratio_df.columns[col_n]] = ratio_df[ratio_df.columns[col_n - 1]]

        ratio_df = ratio_df / 100

        new_cols = [col for col in negomv.columns if col in mes]
        negomv = negomv[new_cols]
        Topten_to_float_ashare = ratio_df * negomv

        Topten_to_float_ashare = Topten_to_float_ashare.dropna(how='all', axis=1)

        # 得到环比改变数据
        delta_tmp = copy.deepcopy(Topten_to_float_ashare)
        to_del_tmp = [col for col in delta_tmp.columns if col.month not in [1, 4, 7, 10]]

        delta_tmp = delta_tmp.drop(to_del_tmp, axis=1)
        delta_tmp = delta_tmp - delta_tmp.shift(1, axis=1)

        delta_ratio = delta_tmp / negomv
        delta_ratio = delta_ratio.fillna(0)

        Delta_to_float_ashare = delta_ratio

        #     pd.DataFrame(0, index=delta_ratio.index,
        #                                      columns=delta_ratio.columns)
        # Delta_to_float_ashare[delta_tmp.columns] = delta_ratio

        # 对于空余的月份，直接复制前值
        for col_n in range(1, len(Delta_to_float_ashare.columns)):
            if Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n]].sum() == 0 and \
                    Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n - 1]].sum() != 0:
                Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n]] = \
                    Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n - 1]]

        res_dict = {'Topten_to_float_ashare': Topten_to_float_ashare.fillna(0),
                    'Delta_to_float_ashare': Delta_to_float_ashare.fillna(0)
                    }

        return res_dict


def compute_factor(status):
    fc = Factor_Compute(status)
    debug_fn = []
    factor_names = [k for k in Factor_Compute.__dict__.keys() if not k.startswith('_')]
    for f in factor_names:
        print(f)
        if f == 'compute_pct_chg_nm':
            res = fc.compute_pct_chg_nm
            if isinstance(res, pd.DataFrame):
                fc.save(res, 'pct_chg_nm'.upper())
        elif f == 'compute_pct_chg_nw':
            res = fc.compute_pct_chg_nw
            if isinstance(res, pd.DataFrame):
                fc.save(res, 'pct_chg_nw'.upper())
        else:
            try:
                tmp = eval('fc.' + f)
                if isinstance(tmp, pd.DataFrame):            # 返回None，表示无需更新
                    fc.save(tmp, f.upper())
                elif isinstance(tmp, dict):
                    for k, v in tmp.items():
                        fc.save(v, k.upper())
                elif not tmp:
                    continue
            except Exception as e:
                print('debug')
                print('有问题的因子名称为{}'.format(f))
                debug_fn = debug_fn + [f]

    if len(debug_fn) > 0:
        print('有问题的因子有：{}'.format(debug_fn))


if __name__ == "__main__":
    # compute_factor('update')

    # 'all'  'update'
    # 测试某个因子
    fc = Factor_Compute('update')    # ('update')
    res = fc.is_open1
    fc.save(res, 'is_open1'.upper())

    Topten_to_float_ashare = res["Topten_to_float_ashare"]
    Delta_to_float_ashare = res["Delta_to_float_ashare"]

    fc.save(Topten_to_float_ashare, 'Topten_to_float_ashare'.upper())
    fc.save(Delta_to_float_ashare, 'Delta_to_float_ashare'.upper())

    panel_path1 = os.path.join(root_dair, '因子预处理模块', '因子')
    add_to_panels(Topten_to_float_ashare, panel_path1, 'Topten_to_float_ashare', freq_in_dat='M')
    add_to_panels(Delta_to_float_ashare, panel_path1, 'Delta_to_float_ashare', freq_in_dat='M')

    panel_path1 = os.path.join(root_dair, '因子预处理模块', '因子（已预处理）')
    add_to_panels(Topten_to_float_ashare, panel_path1, 'Topten_to_float_ashare', freq_in_dat='M')
    add_to_panels(Delta_to_float_ashare, panel_path1, 'Delta_to_float_ashare', freq_in_dat='M')

    # fc.save(res3, 'Long_mom'.upper())

    # data = Data()
    # LONG_MOM = data.LONG_MOM
    # panel_path1 = os.path.join(root_dair, '因子预处理模块', '因子')
    # add_to_panels(LONG_MOM, panel_path1, 'Long_mom', freq_in_dat='M')
    #
    # panel_path2 = os.path.join(root_dair, '因子预处理模块', '因子（已预处理）')
    # add_to_panels(LONG_MOM, panel_path2, 'Long_mom', freq_in_dat='M')

    # 长端动量因子无需做中性化处理，因为这个因子的目的就是选出一些长期缓慢上涨的股票。

    # fc.save(res3, 'compute_pct_chg_nw'.upper())

    # MACD_DIFF_DEA()
    # RSI()
    # PSY()
    # BIAS()

    # fc.save(res, 'Indmom'.upper())
    # panel_path = os.path.join(root_dair, '因子预处理模块', '因子')
    # add_to_panels(res, panel_path, 'Indmom', freq_in_dat='M')

    # reverse_20 = res["M_reverse_20"]
    # reverse_60 = res["M_reverse_60"]
    # reverse_180 = res["M_reverse_180"]
    #
    # fc.save(reverse_20, 'M_reverse_20'.upper())
    # fc.save(reverse_60, 'M_reverse_60'.upper())
    # fc.save(reverse_180, 'M_reverse_180'.upper())
    #
    # panel_path = os.path.join(root_dair, '因子预处理模块', '因子')
    # add_to_panels(reverse_20, panel_path, 'M_reverse_20', freq_in_dat='M')
    # add_to_panels(reverse_60, panel_path, 'M_reverse_60', freq_in_dat='M')
    # add_to_panels(reverse_180, panel_path, 'M_reverse_180', freq_in_dat='M')

    # 每日更新RPS数据时，要先更新 data_management里的价格和贴现率数据
    # fc = Factor_Compute('update')  # ('update')
    # # res = fc.Rps
    # # fc.save(res, 'Rps'.upper())
    #
    # res = fc.compute_pct_chg_nw
    # fc.save(res, 'pct_chg_nw'.upper())

    # res = fc.Rps_by_industry
    # fc.save(res, 'Rps_by_industry'.upper())
    # panel_path = os.path.join(root_dair, '因子预处理模块', '因子')
    # add_to_panels(res, panel_path, 'Rps_by_industry', freq_in_dat='M')
    # grossprofitmargin_q
    # grossprofitmargin_q_diff


