import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from utility.index_enhance import get_factor
from barra_cne6.barra_template import Data


def get_equal_wei_with_index_ret(est_stock_rets, bestN_stock=None, est_indus_rets=None, bestN_indux=None):
    if isinstance(est_indus_rets, pd.DataFrame):
        # 等权配置预测表现好的行业
        wt_indus = pd.DataFrame(np.zeros(np.shape(est_indus_rets)), index=est_indus_rets.index,
                                columns=est_indus_rets.columns)
        # 逐列遍历
        for col, itmes in est_indus_rets.iteritems():
            itmes = itmes.sort_values(ascending=False)
            select_id = itmes.index[0:bestN_indux]
            wt_indus.loc[select_id, col] = 1
        wt_indus.columns = pd.to_datetime(wt_indus.columns)

        # 把二级行业与股票预测数据合并
        si = get_factor(['second_industry'],
                        basic_path=r'D:\pythoncode\IndexEnhancement\多因子选股\正交后因子\因子矩阵')['second_industry']
        est_indus_rets.columns = pd.to_datetime(est_indus_rets.columns)
        si.columns = pd.to_datetime(si.columns)

        # 日期对齐
        dl = list(set(est_indus_rets.columns) & set(est_stock_rets.columns) & set(si.columns))
        dl = sorted(dl)

        bestN_stock_each_indus = int(bestN_stock/bestN_indux)

        wt_stock = pd.DataFrame()
        for date in dl:
            stock_bd = pd.concat([est_stock_rets[date], si[date]], axis=1)
            stock_bd = stock_bd.dropna(how='any')
            stock_bd.columns = ['ext_rets', 'second_indus']

            indus_wei = wt_indus[date]
            grouped = stock_bd.groupby('second_indus')
            wei_tmp = pd.DataFrame()

            for name, group in grouped:
                # 未配置该行业
                if indus_wei[name] == 0:
                    wei_ttmp = pd.DataFrame(np.zeros([group.shape[0], 1]), index=group.index, columns=[date])
                elif indus_wei[name] > 0:
                    if group.shape[0] <= bestN_stock_each_indus:
                        wei_ttmp = (1 / (bestN_stock_each_indus * group.shape[0])) * \
                                   pd.DataFrame(np.ones([group.shape[0], 1]), index=group.index, columns=[date])
                    else:
                        itmes = group['ext_rets'].sort_values(ascending=False)
                        select_id = itmes.index[0:5]
                        wei_ttmp = pd.DataFrame(np.zeros([group.shape[0], 1]), index=group.index, columns=[date])
                        wei_ttmp.loc[select_id, date] = 1 / bestN_stock

                wei_tmp = pd.concat([wei_tmp, wei_ttmp], axis=0)

            wei_tmp = wei_tmp/np.sum(wei_tmp)
            wt_stock = pd.concat([wt_stock, wei_tmp], axis=1)
            wt_stock = wt_stock.fillna(0)
    else:

        wt_stock = pd.DataFrame()
        for col, itmes in est_stock_rets.iteritems():
            wei_tmp = pd.DataFrame(index=est_stock_rets.index)
            items = itmes.sort_values(ascending=False)
            not_nan_stock = items.index[~pd.isna(items)]
            nan_stock = items.index[pd.isna(items)]
            # 非nan的股票数量
            items = items[not_nan_stock]
            # 选择出来的数量
            s_num = int(len(items)*bestN_stock)
            select_id = itmes.index[0:s_num]
            wei_tmp.loc[select_id, col] = 1
            wei_tmp = wei_tmp.fillna(0)
            wei_tmp = wei_tmp/np.sum(wei_tmp)
            wt_stock = pd.concat([wt_stock, wei_tmp], axis=1)

    return wt_stock


def get_equal_wei(est_stock_rets, bestN_stock=20):
    wt_stock = pd.DataFrame()
    for col, items in est_stock_rets.iteritems():
        wei_tmp = pd.DataFrame(index=est_stock_rets.index)
        items = items.sort_values(ascending=False)
        not_nan_stock = items.index[~pd.isna(items)]
        # 非nan的股票数量
        items = items[not_nan_stock]
        # 选择出来的数量
        select_id = items.index[0:bestN_stock]
        wei_tmp.loc[select_id, col] = 1
        wei_tmp = wei_tmp.fillna(0)
        wei_tmp = wei_tmp/np.sum(wei_tmp)
        wt_stock = pd.concat([wt_stock, wei_tmp], axis=1)

    return wt_stock




def select_stock_ToF(est_stock_rets, bestN_stock=100, industry_demand=None):

    res = pd.DataFrame()
    for col, items in est_stock_rets.iteritems():

        items = items.sort_values(ascending=False)
        not_nan_stock = items.index[~pd.isna(items)]
        nan_stock = items.index[pd.isna(items)]
        # 非nan的股票数量
        items = items[not_nan_stock]
        # 选择出来的数量
        select_id = items.index[0:bestN_stock-1]
        tmp_df = pd.DataFrame([True for i in range(0, len(select_id))], index=select_id, columns=[col])
        res = pd.concat([res, tmp_df], axis=1)

    res.fillna(False, inplace=True)

    return res



def target_fun(wei, c):
    '''
    :param wei:  要求的权重，wei是1行N列的array
    :param c:  已知的参数，wei是N行1列的array
    :return:
    '''
    # where x is an 1 - D array with shape(n, ) and args is a tuple of the fixed parameters
    # needed to completely specify the function.
    value = -1 * np.mat(wei) * np.mat(c)
    return value*1e5


def con_f(wei, A_eq, b_eq):
    '''
    wei是1行M列的array
    b_eq为N行1列
    A_eq为N行M列
    '''
    value = np.mat(A_eq) * np.mat(wei).T - np.mat(b_eq)
    return np.sum(value)


def con_f_n(wei, n):
    value = np.sum(wei > 0.0) - n
    return value*0.00001


def get_con(args):
    # 约束条件 分为eq 和ineq
    # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
    n, A_eq, b_eq = args
    con1 = {'type': 'eq', 'fun': con_f_n, 'args': [n]}
    con2 = {'type': 'eq', 'fun': con_f, 'args': [A_eq, b_eq]}
    con3 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    cons = ([con2, con3])
    return cons


# 针对预测收益高的行业，等权配置股票
def optimization_equalwei(est_stock_rets, est_indus_rets, bestN):

    # 等权配置预测表现好的行业
    wt_indus = pd.DataFrame(np.zeros(np.shape(est_indus_rets)), index=est_indus_rets.index, columns=est_indus_rets.columns)
    # 逐列遍历
    for col, itmes in est_indus_rets.iteritems():
        itmes = itmes.sort_values(ascending=False)
        select_id = itmes.index[0:bestN]
        wt_indus.loc[select_id, col] = 1 / bestN
    wt_indus.columns = pd.to_datetime(wt_indus.columns)

    # 把二级行业与股票预测数据合并
    si = get_factor(['second_industry'],
                    basic_path=r'D:\pythoncode\IndexEnhancement\多因子选股\正交后因子\因子矩阵')['second_industry']
    est_indus_rets.columns = pd.to_datetime(est_indus_rets.columns)
    si.columns = pd.to_datetime(si.columns)

    # 日期对齐
    dl = list(set(est_indus_rets.columns) & set(est_stock_rets.columns) & set(si.columns))
    dl = sorted(dl)

    stock_num = 100

    for date in dl:
        date = dl[0]
        stock_bd = pd.concat([est_stock_rets[date], si[date]], axis=1)
        stock_bd = stock_bd.dropna(how='any')
        stock_bd.columns = ['ext_rets', 'second_indus']
        stock_bd = pd.get_dummies(stock_bd)
        stock_bd.columns = ['ext_rets'] + [col.split('_')[-1] for col in stock_bd.columns if 'second' in col]

        c = stock_bd['ext_rets'].values.reshape(len(stock_bd), 1)
        x0 = np.zeros(len(stock_bd)) / len(stock_bd)  # 初始值

        # c.shape
        bnds = []
        for j in range(0, len(stock_bd)):
            bnds.append((0.0, 1.0))
        bnds = tuple(bnds)

        A_eq = stock_bd[list(stock_bd.columns[1:])].values.T
        #A_eq.shape
        # 对齐
        indus_exposure = wt_indus[date]
        indus_exposure = indus_exposure.reindex(list(stock_bd.columns[1:]))
        indus_exposure = indus_exposure.fillna(0)
        b_eq = indus_exposure.values.reshape(-1, 1)
        # b_eq.shape
        args = [stock_num, A_eq, b_eq]
        cons = get_con(args)

        res = minimize(target_fun, x0, args=c, bounds=bnds, constraints=cons, options={'disp': True})
        wei = res.x
        print(res.success)

        res.status
        res.message
        res.nit

        wei = wei / np.sum(wei)
        cur_wt = pd.DataFrame(wei, index=stock_bd.index, columns=[date])




def optimization(est_stock_rets, index_wei, est_indus_rets=None):
    '''
    :param est_indus_rets:   预测的股票收益率，若有，则是超配低配模式，若没有，则是行业中性模式
    :param est_stock_rets:   预测的行业收益率
    :param index_wei:        指数的行业权重
    :return:                 股票各期权重
    '''

    if isinstance(est_indus_rets, pd.DataFrame):
        # 获取股票的二级行业情况数据
        si = get_factor(['second_industry'],
                        basic_path=r'D:\pythoncode\IndexEnhancement\多因子选股\正交后因子\因子矩阵')['second_industry']

        est_indus_rets.columns = pd.to_datetime(est_indus_rets.columns)
        si.columns = pd.to_datetime(si.columns)

        dl = list(set(est_indus_rets.columns) & set(est_stock_rets.columns) & set(index_wei.columns))
        dl = sorted(dl)
    else:
        # 获取股票的一级行业情况数据
        si = get_factor(['industry_zx'],
                        basic_path=r'D:\pythoncode\IndexEnhancement\多因子选股\正交后因子\因子矩阵')['industry_zx']
        est_indus_rets.columns = pd.to_datetime(est_indus_rets.columns)
        si.columns = pd.to_datetime(si.columns)

        dl = list(set(est_stock_rets.columns) & set(index_wei.columns))
        dl = sorted(dl)

    total_wei = pd.DataFrame()
    for date in dl:
        if isinstance(est_indus_rets, pd.DataFrame):
            # 超配低配模式下的行业暴露
            indus_bd = pd.concat([est_indus_rets[date], index_wei[date]], axis=1)
            indus_bd = indus_bd.dropna(how='any')
            indus_bd.columns = ['est_rets', 'index_wet']
            up_quantile = 25
            expand_beta = 2
            dow_quantile = 70
            shrink_beta = 0.5
            indus_exposure = adjust_weight(indus_bd, up_quantile, expand_beta, dow_quantile, shrink_beta)
            # pd.isna(indus_exposure)
        else:
            # 超配低配模式下的行业暴露就是指数的行业暴露
            indus_exposure = index_wei[date]

        stock_bd = pd.concat([est_stock_rets[date], si[date]], axis=1)
        stock_bd = stock_bd.dropna(how='any')
        stock_bd.columns = ['ext_rets', 'second_indus']
        stock_bd = pd.get_dummies(stock_bd)
        stock_bd.columns = ['ext_rets'] + [col.split('_')[-1] for col in stock_bd.columns if 'second' in col]

        c = stock_bd['ext_rets'].values.reshape(1, len(stock_bd))
        x0 = np.ones(len(stock_bd)) / len(stock_bd)   # 初始值

        c.shape
        bnds = []
        for j in range(0, len(stock_bd)):
            bnds.append((0, 1))
        bnds = tuple(bnds)

        stock_num = 100

        A_eq1 = stock_bd[list(stock_bd.columns[1:])].values.T
        [a, b] = A_eq1.shape
        tmp = np.ones([1, b])
        A_eq = np.r_[A_eq1, tmp]
        A_eq.shape
        # 对齐
        indus_exposure = indus_exposure.reindex(list(stock_bd.columns[1:]))
        indus_exposure = indus_exposure.fillna(0)
        b_eq = indus_exposure.values.reshape(-1, 1)
        b_eq = np.r_[b_eq, np.array(1).reshape(1, 1)]
        b_eq.shape
        args = [stock_num, A_eq, b_eq]
        cons = get_con(args)

        res = minimize(target_fun, x0, args=c, bounds=bnds,  constraints=cons)
        wei = res.x
        wei = wei / np.sum(wei)
        cur_wt = pd.DataFrame(wei, index=stock_bd.index, columns=[date])

    total_wei = pd.concat([total_wei, cur_wt], axis=1)

    return total_wei


def adjust_weight(ori_pd, up_quantile, expand_beta, dow_quantile, shrink_beta):
    '''
    :param ori_pd:  原始数据, dataframe格式，第一列为预测收益，第二列为指数权重
    :param up_quantile:  前多少分位数
    :param expand_beta:  在指数权重的基础上乘以该系数
    :param dow_quantile:  后多少分位数
    :param shrink_beta:  在指数权重的基础上乘以该系数
    :return:
    '''
    ori_pd = ori_pd.sort_values('est_rets', ascending=False)

    loc_up = np.percentile(range(0, len(ori_pd['est_rets'])), up_quantile)
    loc_down = np.percentile(range(0, len(ori_pd['est_rets'])), dow_quantile)

    ori_pd['num'] = range(0, len(ori_pd))
    for ind in ori_pd.index:
        if ori_pd.loc[ind, 'num'] < loc_up:
            # 对本来就是权重较高的行业，就不再增加权重了
            if ori_pd.loc[ind, 'index_wet'] < 4:
                ori_pd.loc[ind, 'exposure'] = expand_beta * ori_pd.loc[ind, 'index_wet']
            else:
                ori_pd.loc[ind, 'exposure'] = ori_pd.loc[ind, 'index_wet']
        elif ori_pd.loc[ind, 'num'] > loc_down:
            ori_pd.loc[ind, 'exposure'] = shrink_beta * ori_pd.loc[ind, 'index_wet']
        else:
            ori_pd.loc[ind, 'exposure'] = ori_pd.loc[ind, 'index_wet']

    ori_pd['exposure'] = ori_pd['exposure']/np.sum(ori_pd['exposure'])

    return ori_pd['exposure']


# 把code列换成其他的信息
def code2other(base_df, code_list, target):
    if 'code' not in base_df.columns:
        return print('列名中没有code')
    if target not in base_df.columns:
        return print('列名中没有{}'.format(target))

    index_l = [i for i in base_df.index if base_df.loc[i, 'code'] in code_list]
    value = base_df.loc[index_l, target].values
    return value


def new_record_stock(stock_wei, save_name='每期', save_path=None):

    data = Data()
    all_stocks_code = data.all_stocks_code
    all_stocks_code = all_stocks_code[['wind_code', 'sec_name']]
    all_stocks_code = all_stocks_code.set_index('wind_code')

    maxl = np.sum(stock_wei != 0.0, axis=0).max()
    res_to_csv = pd.DataFrame(index=range(0, maxl))

    for col, items in stock_wei.iteritems():
        selected = items[items.index[items != 0]]
        selected = all_stocks_code.loc[selected.index, :]
        selected = pd.DataFrame(selected.values, columns=[col])
        res_to_csv = pd.concat([res_to_csv, selected], axis=1)

    if not save_path:
        save_path = r'D:\pythoncode\IndexEnhancement'
    res_to_csv.to_csv(os.path.join(save_path, save_name + '选股结果.csv'), encoding='gbk')
    return res_to_csv


def record_stock_and_industry(stock_wei, est_second_indus_rets, best_n):
    fp = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'

    stock_wei2csv = pd.DataFrame()
    for col in stock_wei.columns:
        tmp_panel_wei = stock_wei[col]
        datestr = col.to_pydatetime().strftime("%Y-%m-%d")
        datdf = pd.read_csv(os.path.join(fp, datestr + '.csv'), encoding='gbk', engine='python')
        tmp_panel_wei = tmp_panel_wei.drop(tmp_panel_wei.index[tmp_panel_wei == 0.0])
        name_list = code2other(datdf, tmp_panel_wei.index, 'name')
        s_i = code2other(datdf, tmp_panel_wei.index, 'second_industry')

        tmp_panel_wei_pd = pd.DataFrame({datestr + ':code': tmp_panel_wei.index,
                                         datestr + ':name': name_list,
                                         datestr + ':weight': tmp_panel_wei.values,
                                         datestr + ':indus': s_i
                                         })
        stock_wei2csv = pd.concat([stock_wei2csv, tmp_panel_wei_pd], axis=1)
    savepath = r'D:\pythoncode\IndexEnhancement'
    stock_wei2csv.to_csv(os.path.join(savepath, '每期行业轮动股票结果.csv'), encoding='gbk')

    indus2csv = pd.DataFrame()
    # 逐列遍历
    for col, itmes in est_second_indus_rets.iteritems():
        itmes = itmes.sort_values(ascending=False)
        select_id = itmes.index[0:best_n]
        tmp_i = pd.DataFrame(select_id.values, columns=[col])

        indus2csv = pd.concat([indus2csv, tmp_i], axis=1)
    indus2csv.to_csv(os.path.join(savepath, '每期前{}行业结果.csv'.format(best_n)), encoding='gbk')



