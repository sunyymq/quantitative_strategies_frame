import pandas as pd
import numpy as np
import os
import copy
from datetime import datetime
from utility.tool3 import adjust_months, append_df
from utility.stock_pool import scopy_condition, rise_condition
from utility.tool0 import Data
from utility.single_factor_test import Backtest_stock
from utility.index_enhance import get_matrix_data
from utility.stock_pool import month_return_compare_to_market_index,  get_wei, name_to_code, code_to_name
from utility.constant import data_dair, root_dair
from utility.analysis import BackTest


save_path = r'D:\Database_Stock\临时'


def adjust_num(d_df, max_num):
    data = Data()
    roattm = data.ROA_TTM

    for col, items in d_df.iteritems():
        if np.sum(items==True) > max_num:
            tmp_df = pd.concat([items, roattm[col]], axis=1)
            tmp_df.columns = ['t_or_f', 'roettm']
            tmp_df = tmp_df.sort_values('roettm', ascending=False)
            tmp_df['t_or_f_cum'] = tmp_df['t_or_f'].cumsum()
            tmp_df.index[tmp_df['t_or_f_cum'] > max_num]
            items[tmp_df['t_or_f_cum'] > max_num] = False

    return d_df


def roe_stock_pool(max_hold_num, special_plate=None, analyse_indus=True,
                   save=True):
    '''

    :param max_hold_num:  最大持有股票数量
    :param special_plate:  是否针对特定板块
    :param save:  是否储存每期的选股结果
    :param shift:  是否向右移动一期
    :return:
    '''
    # 基础资料
    data = Data()
    stock_basic = data.stock_basic_inform
    firstindustry = stock_basic['中信一级行业']

    all_stocks_code = stock_basic[['SEC_NAME', 'IPO_DATE']]
    all_stocks_code['IPO_DATE'] = pd.to_datetime(all_stocks_code['IPO_DATE'])

    # roe
    roettm = data.roettm
    # 净利润同比增速
    netprofit = data.netprofitgrowrate
    # 销售毛利率
    grossincome = data.grossincomeratiottm
    # 资产负债率
    debtassetsratio = data.debtassetsratio
    # 经营活动产生的现金流量净额/营业收入_TTM(%)
    cashrateofsalesttm = data.cashrateofsalesttm
    # 估值
    ep = data.EP
    pe = ep.applymap(lambda x: 1/x)

    if not special_plate:    # 未对板块做出要求

        # 条件
        cond_roe_scope = scopy_condition(roettm, minV=15, maxV=60)
        cond_roe_rise = rise_condition(roettm, 1)
        cond_netprofit_rise = rise_condition(netprofit, 2)
        cond_grossincome_rise = rise_condition(grossincome, 1)
        cond_debtratio_scope = scopy_condition(debtassetsratio, maxV=60)
        cond_cashrate_scope = scopy_condition(cashrateofsalesttm, minV=0.1)  # 条件不能为0，设定一个稍微大于0的数

        # cond_total = cond_roe_scope & cond_roe_rise & cond_netprofit_rise & \
        #              cond_grossincome_rise & cond_debtratio_scope & cond_cashrate_scope

        cond_total = cond_roe_scope & cond_roe_rise & cond_netprofit_rise & cond_cashrate_scope & cond_grossincome_rise \
                     & cond_debtratio_scope
        cond_total.drop(cond_total.columns[np.sum(cond_total == True, axis=0) == 0], axis=1, inplace=True)

        del_industry_list = ['有色金属', '钢铁', '非银金融', '非银行金融']
        # 剔除部分行业
        for col, items in cond_total.iteritems():
            for i in items.index:
                if items[i]:
                    if i in firstindustry.index:
                        if firstindustry[i] in del_industry_list:
                            items[i] = False

        # 剔除上市未满N年得股票，N = 2
        del_num = 0
        N = 1
        for col, items in cond_total.iteritems():
            for i in items.index:
                if items[i]:
                    if i in all_stocks_code.index:
                        de = col - all_stocks_code.loc[i, 'ipo_date'.upper()]
                        if de.days < N * 365:
                            items[i] = False
                            del_num = del_num + 1
        print('通过剔除新股条件，剔除得股票次数为{}'.format(del_num))

        # 剔除st
        for col, items in cond_total.iteritems():
            for i in items.index:
                if i in all_stocks_code.index and 'ST' in all_stocks_code.loc[i, 'sec_name'.upper()]:
                    items[i] = False

        # 把公告日期调整为实际交易日期
        cond_total = adjust_months(cond_total)
        cond_total.sum()
        # 添加PE限制
        cond_total_appended = append_df(cond_total)
        cond_pe = scopy_condition(pe, maxV=100)
        cond_total_appended = cond_total_appended & cond_pe
        cond_total_appended.drop(cond_total_appended.columns[np.sum(cond_total_appended == True, axis=0) == 0], axis=1,
                                 inplace=True)

        # 调整股票数量
        tmp = np.sum(cond_total_appended == True, axis=0).max()
        print('每期满足条件的股票的最大数量为{}'.format(tmp))
        cond_adjusted = adjust_num(cond_total_appended, max_hold_num)

        # to_save = list(cond_adjusted.index[cond_adjusted[cond_adjusted.columns[-1]] == True])
        # add_stock_pool_txt(to_save, 'ROE选股策略', renew=True)

    if analyse_indus:
        all_first_industry = firstindustry.drop_duplicates()
        selected_first_industry_num = pd.Series(np.zeros(len(all_first_industry)), index=all_first_industry.values)
        total_num = np.sum(cond_total == True, axis=0).sum()

        # 统计每期选的一级行业的公司数量
        for col, items in cond_adjusted.iteritems():
            tmp_pd = pd.DataFrame({'wei': items, 'ind': firstindustry}, index=items.index)
            tmp_pd.dropna(how='any', axis=0, inplace=True)
            selected = tmp_pd.loc[tmp_pd['wei']==True, 'ind']
            for v in selected.values:
                selected_first_industry_num[v] = selected_first_industry_num[v] + 1

        selected_first_industry_num.to_csv(os.path.join(save_path, 'roe选股行业统计.csv'), encoding='gbk')
        selected_num = np.sum(cond_adjusted == True, axis=0)
        selected_num = pd.DataFrame(selected_num)
        selected_num.to_csv(os.path.join(save_path, 'roe选股每期数量.csv'), encoding='gbk')

    # 用于回测的条件选择
    cond_adjusted = cond_adjusted.shift(1, axis=1)

    if save:
        # 存储每次更新财务数据时的结果
        maxl = np.sum(cond_adjusted ==True, axis=0).max()
        res_to_csv = pd.DataFrame(index=range(0, maxl))

        # 对选出来的股票向后移动一期，因为回测是根据月度收益模式，日期为月末最后一天，而公告日期为月末最后一天，
        # 所有需要向后移动一期。
        for col, items in cond_adjusted.iteritems():
            selected = items[items.index[items == True]]
            selected = all_stocks_code.loc[selected.index, 'sec_name'.upper()]
            selected = pd.DataFrame(selected.values, columns=[col])

            res_to_csv = pd.concat([res_to_csv, selected], axis=1)

        res_to_csv.to_csv(os.path.join(save_path, '每期roe选股结果.csv'), encoding='gbk')

    print('股票池确定完毕')

    cond_adjusted.dropna(how='all', axis=1, inplace=True)
    wei = get_wei(cond_adjusted)
    res = wei

    return res


def easy_month_bt(wei):

    bt = BackTest(wei, 'M', fee_type='No_fee', benchmark_str='WindA', hedge_status=False)
    bt.run_bt()
    ana = bt.analysis()
    bt.plt_pic()
    sp = os.path.join(root_dair, '临时', 'ROE策略表现.csv')
    ana.to_csv(sp, encoding='gbk')
    print('回测结束, 进行回测结果分析...')


def stock_list_special_month(special_date):
    stock_pool = pd.read_csv(os.path.join(save_path, '每期roe选股结果.csv'), encoding='gbk')
    stock_pool.set_index(stock_pool.columns[0], inplace=True)

    stock_pool.columns = pd.to_datetime(stock_pool.columns)
    ff = None
    for c in stock_pool.columns:
        if c.year == special_date.year and c.month == special_date.month:
            ff = c
            break

    tmp = list(stock_pool[ff].dropna())
    codes_list = name_to_code(tmp)
    return codes_list


def update_roe_stock_strategy(per_pre_month, per_month):
    # wei = roe_stock_pool(max_hold_num=50, plate='电子')
    # wei_for_rqalpha = change_code_f_rqalpha(wei)
    # easy_month_bt(wei)

    roe_stock_pool(50, analyse_indus=False,
                   save=True)

    sl = stock_list_special_month(per_pre_month)
    res1, res2 = month_return_compare_to_market_index(sl, per_month)
    save_path = r'D:\pythoncode\IndexEnhancement\股票池_最终'
    res1.to_csv(os.path.join(save_path, 'roe选股上个月组合表现.csv'), encoding='gbk')

    sl2 = stock_list_special_month(per_month)
    # 添加概念
    data = Data()
    concept = data.concept

    tmp = pd.DataFrame(data=code_to_name(sl2), index=sl2, columns=['name'])
    res_df = pd.concat([tmp, concept], axis=1, join='inner')

    for k, v in res_df['CONCEPT'].items():
        try:
            res_df.loc[k, 'CONCEPT'] = v.replace('[', '').replace(']', '').replace('\'', '')
        except Exception:
            pass

    res_df.to_csv(os.path.join(save_path, 'roe选股当月组合.csv'), encoding='gbk')


def bt_roe_stock_strategy(st=None, ed=None):
    wei = roe_stock_pool(max_hold_num=50)
    if ed and st:
        new_col = [col for col in wei.columns if st <= col <= ed]
    elif ed:
        new_col = [col for col in wei.columns if col <= ed]
    elif st:
        new_col = [col for col in wei.columns if st <= col]

    wei = wei[new_col]
    easy_month_bt(wei)


if __name__ == '__main__':

    mode = 'follow'
    # 跟踪
    if mode == 'follow':
        tod = datetime.today()
        per_month = datetime(tod.year, tod.month - 1, 1)
        per_pre_month = datetime(tod.year, tod.month - 2, 1)
        update_roe_stock_strategy(per_pre_month, per_month)
    # 回测
    if mode == 'bt':
        start_months = datetime(2020, 1, 1)
        end_months = datetime(2020, 6, 30)
        bt_roe_stock_strategy(start_months, end_months)




