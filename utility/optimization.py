import re
import os
import numpy as np
import pandas as pd
import time
import cvxpy as cp
import winsound
import matplotlib.pyplot as plt
from functools import reduce
from collections import defaultdict
from seaborn import heatmap
from copy import deepcopy
from scipy.optimize import linprog as lp
from cvxopt import solvers, matrix
from utility.single_factor_test import Backtest_stock
from utility.factor_data_preprocess import info_cols
from utility.tool0 import Data


def select_import_wei(wei_se, n_max, abs_max_ratio=0.9, max_in_indus=2):
    '''
    abs_max = 15        留下绝对数前15的个股权重
    max_in_indus = 2    留下行业内排名前2的个股权重
    同时删除部分权重过小的股票，只留下极少部分股票做进一步的优化，删除第一部优化时权重小于0.001的股票。
    '''
    data = Data()
    basic_inform = data.stock_basic_inform
    indus_map = basic_inform.loc[wei_se.index, '申万一级行业']
    # 强制保留w1中各行业最大权重股以及其他权重前15的股票
    res_wei = pd.Series(index=wei_se.index)

    # 分行业里面权重大的留下
    wei_se = wei_se.sort_index()
    dat_df = pd.DataFrame({'wei': wei_se, 'industry': indus_map})
    grouped = dat_df.groupby('industry')
    for ind, v in grouped:
        tmp = v['wei'].sort_values(ascending=False)
        res_wei[tmp[:max_in_indus].index] = tmp[:max_in_indus]

    num_0 = np.sum(res_wei > 0)
    # 分行业留下权重前2的股票之外，在找到其余的股票中绝对权重排名前 (n_max * abs_max_ratio - 已经留下的股票数量)的股票数量。
    num_1 = int(n_max * abs_max_ratio) - num_0
    tmp = list(set(wei_se.index) - set(res_wei.dropna().index))
    left_se = wei_se[tmp]
    left_se = left_se.sort_values(ascending=False)
    tmp1 = left_se[:num_1]
    res_wei[tmp1.index] = tmp1

    res_wei = res_wei.dropna()
    n_left = np.sum(res_wei > 0)
    # np.sum(res_wei)
    if n_left > n_max:
        print('留下的股票过多，重新选择')
        input("暂时挂起.... ")

    tmp2 = wei_se[wei_se > 0.001]
    tobe_opt = [i for i in tmp2.index if i not in res_wei.index]

    return res_wei, n_left, tobe_opt


# 根据变量和条件，生成约束列
def generates_constraints(para_dict, con_dict):
    '''
        个股权重约束：            1，不为负，nonneg=True
                                 2，最大值, x <= max_wei
                                    最小值， x >= min_wei,
        成份股权重和约束（如有）：  is_in_bench.T * x >= in_benchmark_wei
        行业中性约束:             dum @ x == ind_wei,
        行业非中性但控制行业约束：dum @ x <= ind_wei + industry_max_expose,
                                 dum @ x >= ind_wei - industry_max_expose,
                                 cp.sum(x) == 1,
        行业不做任何约束：        cp.sum(x) == 1,
        风险因子约束：           limit_df = limit_factor_df[key]
                                # 调整到同一index排序
                                limit_df = limit_df[wei_tmp.index]
                                limit_f = limit_df.T.values
                                bench_limit_expo = np.dot(limit_df.T, wei_tmp)
            1，如果风险因子完全不暴露： limit_f.T * x = bench_limit_expo
            2，风险因子暴露一定数量：   limit_f.T * x >= bench_limit_expo - value
                                       limit_f.T * x <= bench_limit_expo - value
        换手率约束：             cvx.norm(x - pre_x, 1) <= 0.3
        跟踪误差约束：           cp.quad_form(x - wei_tmp, P) <= te
        最大股票数量约束：        y = cp.Variable(len(ret_tmp), boolean=True)
                                 x - y <= 0,
                                 cp.sum(y) <= 120,
        '''

    x = para_dict['x']
    if 'y' in para_dict.keys():
        y = para_dict['y']
        y_sum = para_dict['y_sum']
    else:
        y = None
        y_sum = None

    max_wei = para_dict['max_wei']   # 个股最大权重
    min_wei = para_dict['min_wei']

    in_benchmark_wei = para_dict['in_benchmark_wei']
    is_in_bench = para_dict['is_in_bench']
    dum = para_dict['dum']
    wei_tmp = para_dict['wei_tmp']
    ind_wei = para_dict['ind_wei']
    ret_e = para_dict['ret_e']
    risk_factor_dict = para_dict['risk_factor_dict']
    limit_factor_df = para_dict['limit_factor_df']
    pre_w = para_dict['pre_w']
    P = para_dict['P']
    total_wei = para_dict['total_wei']

    in_benchmark = con_dict['in_benchmark']
    industry_expose_control = con_dict['industry_expose_control']
    industry_max_expose = con_dict['industry_max_expose']
    turnover = con_dict['turnover']
    te = con_dict['te']

    constraints = [x <= max_wei]
    if isinstance(min_wei, np.ndarray):
        constraints.append(x >= min_wei)

    if not in_benchmark:
        constraints.append(is_in_bench.values.T * x >= in_benchmark_wei)

    if industry_expose_control:
        if industry_max_expose == 0:
            constraints.append(dum @ x == ind_wei)
        else:
            constraints.append(dum @ x <= ind_wei + industry_max_expose)
            tmp = ind_wei - industry_max_expose
            tmp[tmp < 0] = 0
            constraints.append(dum @ x >= tmp)
            constraints.append(cp.sum(x) == total_wei)
    else:
        constraints.append(cp.sum(x) == total_wei)

    if len(risk_factor_dict) != 0:
        for key, value in risk_factor_dict.items():
            print(key)
            print(value)
            limit_df = limit_factor_df[key]
            # 调整到同一index排序
            # 第一次优化时，待优化的x的Index与指数权重wei的index相同，
            # 但第二次优化时，待优化的x的Index与指数权重wei的index不同，算指数的风险因子暴露与算待优化的风险因子暴露
            # 的因子矩阵就不一样了。
            limit_df = limit_df[wei_tmp.index]
            bench_limit_expo = np.dot(limit_df.T, wei_tmp)
            limit_f = limit_df[ret_e.index].T.values
            len(ret_e.index)
            if y and value == 0:
                value = 2

            if value == 0:
                constraints.append(limit_f.T * x == bench_limit_expo)
            else:
                constraints.append(limit_f.T * x >= bench_limit_expo - value)
                constraints.append(limit_f.T * x <= bench_limit_expo + value)

    if turnover and isinstance(pre_w, pd.Series) and turnover < 1:
        constraints.append(cp.norm(x - pre_w.values, 1) <= turnover)

    # 当作二次优化的时候，x的数量和指数权重的已经发生很大变化了，而且P也变了。
    # 简化期间，在二次优化时就先忽略这个功能。
    if te and not y:
        constraints.append(cp.quad_form(x - wei_tmp, P) <= te)

    if y:
        constraints.append(x - y <= 0.)
        constraints.append(cp.sum(y) <= y_sum)

    return constraints


def generates_problem(q, x, lam, P, c, pre_w, constraints, te):

    if te:
        if isinstance(pre_w, pd.Series):
            # norm(x, 1) 表示 ∑i|xi|
            prob = cp.Problem(cp.Maximize(q.T * x - c * cp.norm(x-pre_w.values, 1)), constraints)
        else:
            prob = cp.Problem(cp.Maximize(q.T * x), constraints)
    else:
        if isinstance(P, np.ndarray):
            if isinstance(pre_w, pd.Series):
                prob = cp.Problem(cp.Maximize(q.T * x - lam * cp.quad_form(x, P) - c * cp.norm(x - pre_w.values, 1)), constraints)
            else:
                prob = cp.Problem(cp.Maximize(q.T * x - lam * cp.quad_form(x, P)), constraints)
        else:
            # P为nan，lam为0
            if isinstance(pre_w, pd.Series):
                prob = cp.Problem(cp.Maximize(q.T * x - c * cp.norm(x-pre_w.values, 1)), constraints)
            else:
                prob = cp.Problem(cp.Maximize(q.T * x), constraints)
    return prob


def optimization_fun(ret, e, bench_wei, pre_w=None, is_enhance=True, lam=10, c=0.015, turnover=None, te=None,
                     industry_expose_control=True, industry_max_expose=0, control_factor_dict={}, limit_factor_df=None,
                     in_benchmark=True, in_benchmark_wei=0.8, max_num=None, s_max_type='tight'):

    uni_index = list(set(e.index) & set(ret.index))
    uni_index.sort()
    # 选出停牌的股票
    wei_tmp = bench_wei.dropna()
    # 对前期权重和本期成份股权重取并集
    if isinstance(pre_w, pd.Series):
        index_all = list(set(wei_tmp.index) | set(pre_w.index))
        pre_w = pre_w[index_all].fillna(0)
        wei_tmp = wei_tmp[index_all].fillna(0)
    suspend = [i for i in wei_tmp.index if i not in uni_index]
    if isinstance(pre_w, pd.Series):
        sus_holds = [i for i in suspend if i in pre_w.index and pre_w[i] > 0]
        if len(sus_holds) > 0:
            sus_wei = pre_w[sus_holds]
            print('停牌权重为{}'.format(sus_wei.sum()))
            in_benchmark_wei = in_benchmark_wei - sus_wei.sum()
            max_num = max_num + len(sus_wei)

            if sus_wei.sum() > 0.4:
                te = None
                lam = 0
                turnover = None
                control_factor_dict = {}

            if turnover:
                turnover = turnover + sus_wei.sum()

            if te:
                te = te + sus_wei.sum()
        else:
            sus_wei = None
    else:
        sus_wei = None

    if in_benchmark:
        # 剔除停牌的股票后的指数成分权重
        wei_tmp = wei_tmp.drop(suspend)
        if isinstance(pre_w, pd.Series):
            pre_w = pre_w.drop(suspend)

        e_tmp = e.loc[wei_tmp.index, wei_tmp.index].fillna(0)
        ret_tmp = ret[wei_tmp.index].fillna(0)
        is_in_bench = None

    else:
        wei_tmp = wei_tmp.drop(suspend)
        if isinstance(pre_w, pd.Series):
            pre_w = pre_w.drop(suspend)

        # 确保几个重要变量有相同的index
        e_tmp = e.loc[uni_index, uni_index]
        ret_tmp = ret[uni_index]
        wei_tmp = wei_tmp[uni_index].fillna(0)
        if isinstance(pre_w, pd.Series):
            pre_w = pre_w[uni_index].fillna(0)
        # 如果可以选非成份股，则可以确定一个成份股权重比例的约束条件。
        is_in_bench = deepcopy(wei_tmp)
        is_in_bench[is_in_bench > 0] = 1       # 代表是否在成份股内的变量

    data = Data()
    basic = data.stock_basic_inform
    industry_sw = basic[['申万一级行业']]
    # 股票组合的行业虚拟变量
    industry_map = industry_sw.loc[ret_tmp.index, :]

    # dummies_bench = pd.get_dummies(industry_map.loc[bench_wei.index, :])
    # dummies_bench.sum()  不同行业的公司数量
    industry_map.fillna('综合', inplace=True)
    dummies = pd.get_dummies(industry_map[industry_map.columns[0]])

    dummies.sum()

    # 个股最大权重为行业权重的 3/4
    ind_wei = np.dot(dummies.T, wei_tmp)
    ind_wei_se = pd.Series(index=dummies.columns, data=ind_wei)

    if s_max_type == 'loose':
        industry_map['max_wei'] = None
        for i in industry_map.index:
            try:
                industry_map.loc[i, 'max_wei'] = ind_wei_se[industry_map.loc[i, '申万一级行业']]
            except Exception as e:
                industry_map.loc[i, 'max_wei'] = 0.03
        max_wei = industry_map['max_wei'].values
        min_wei = None
    elif s_max_type == 'tight':
        max_wei_tmp = wei_tmp + 0.01
        max_wei = max_wei_tmp.values
        min_wei_tmp = wei_tmp - 0.01
        min_wei_tmp[min_wei_tmp < 0] = 0
        min_wei = min_wei_tmp.values

    x = cp.Variable(len(ret_tmp), nonneg=True)

    q = ret_tmp.values
    P = e_tmp.values

    ind_wei = np.dot(dummies.T, wei_tmp)                    # b.shape
    ind_wei_su = pd.Series(ind_wei, index=dummies.columns)
    ind_wei_su.sum()
    if isinstance(pre_w, pd.Series):
        pre_ind_wei = np.dot(dummies.T, pre_w)  # b.shape
        pre_ind_wei_su = pd.Series(ind_wei, index=dummies.columns)
        test0 = pre_ind_wei_su - ind_wei_su
        test0.abs().sum()

    dum = dummies.T.values                             # A.shape

    # 优化的总权重，若前期持有停牌的，需要先做剔除处理
    if isinstance(sus_wei, pd.Series):
        tw = 1 - sus_wei.sum()
    else:
        tw = 1

    if te:
        te_tmp = (te ** 2) / 12
    else:
        te_tmp = None

    para_dict = {'x': x,
                 'max_wei': max_wei,
                 'min_wei': min_wei,
                 'in_benchmark_wei': in_benchmark_wei,
                 'is_in_bench': is_in_bench,
                 'ret_e': ret_tmp,
                 'dum': dum,
                 'wei_tmp': wei_tmp,
                 'ind_wei': ind_wei,
                 'risk_factor_dict': control_factor_dict,
                 'limit_factor_df': limit_factor_df,
                 'pre_w': pre_w,
                 'P': P,
                 'total_wei': tw,
                 }
    con_dict = {'in_benchmark': in_benchmark,
                'industry_expose_control': industry_expose_control,
                'industry_max_expose': industry_max_expose,
                'turnover': turnover,
                'te': te_tmp,
                }

    constraints = generates_constraints(para_dict, con_dict)
    prob = generates_problem(q, x, lam, P, c, pre_w, constraints, te_tmp)

    # len(q)
    # P.shape
    # len(pre_w)
    # np.sum(pd.isna(P))

    print('开始优化...')
    time_start = time.time()
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception as e:
        print('Solver ECOS failed')

    status = prob.status
    # 如果初始条件无解，需要放松风险因子的约束
    iters = 0
    while status != 'optimal' and iters < 4:
        if iters == 0:
            if len(control_factor_dict) > 0:
                tmp_d = deepcopy(control_factor_dict)
                for k, v in tmp_d.items():
                    tmp_d[k] = v + 2
                para_dict['risk_factor_dict'] = tmp_d

        if iters == 1:
            if turnover and turnover > 1:
                turnover = turnover + 0.2
                con_dict['turnover'] = turnover
            else:
                para_dict['risk_factor_dict'] = {}

        if iters == 2:
            industry_max_expose = industry_max_expose + 0.05
            con_dict['industry_max_expose'] = industry_max_expose

        iters = iters + 1
        constraints = generates_constraints(para_dict, con_dict)
        prob = generates_problem(q, x,  lam, P, c, pre_w, constraints, te)
        print('第{}次优化'.format(iters))
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception as e:
            print('Solver ECOS failed')

        status = prob.status

    time_end = time.time()
    print('优化结束，用时', time_end - time_start)
    print('优化结果为{}'.format(status))

    if not (prob.status == 'optimal' or prob.status == 'optimal_inaccurate'):
        input('input:未得出最优解，请检查')

    # 返回值
    wei_ar = np.array(x.value).flatten()  # wei_ar.size
    wei_se = pd.Series(wei_ar, index=ret_tmp.index)

    # 设定标准，一般情况下无需对股票数量做二次优化，只有股票数量过多是才需要。
    if np.sum(x.value > 0.001) > max_num:
        print('进行第二轮股票数量的优化')
        # wei_selected, n2, tobe_opt = select_import_wei(wei_se, max_num)
        tobe_opt = list(wei_se[wei_se > 0.001].index)
        print('第二次优化为从{}支股票中优化选择出{}支'.format(len(tobe_opt), max_num))

        # 经过处理后，需要优化的计算量大幅度减少。比如第一次优化后，权重大于0.001的股票数量是135，超过最大要求的100。
        # 我们首先保留其中前90，然后从后面的45个中选择10保留下来。
        len(tobe_opt)
        e_tmp2 = e_tmp.loc[tobe_opt, tobe_opt]
        ret_tmp2 = ret_tmp[tobe_opt]
        # wei_tmp2 = wei_tmp[tobe_opt]

        if isinstance(is_in_bench, pd.Series):
            is_in_bench2 = is_in_bench[tobe_opt]
        else:
            is_in_bench2 = None

        dummies2 = pd.get_dummies(industry_map.loc[tobe_opt, industry_map.columns[0]])
        dum2 = dummies2.T.values
        # 小坑
        new_ind = ind_wei_su[dummies2.columns]
        new_ind = new_ind / new_ind.sum()
        ind_wei2 = new_ind.values

        if s_max_type == 'tight':
            max_wei_tmp = wei_tmp + 0.01
            min_wei_tmp = wei_tmp - 0.01
            min_wei_tmp[min_wei_tmp < 0] = 0
            max_wei2 = max_wei_tmp[tobe_opt].values
            min_wei2 = min_wei_tmp[tobe_opt].values

        # # 对个股权重优化的坑，开始时是行业权重乘以0.75，但在二次优化的时候，可能有的行情的权重不够用了。
        # max_wei2 = 3 * industry_map.loc[tobe_opt, 'max_wei'].values

        if pre_w:
            pre_w = pre_w[tobe_opt]

        wei_tmp2 = wei_tmp[tobe_opt]
        limit_factor_df2 = limit_factor_df.loc[tobe_opt, :]

        P2 = lam * e_tmp2.values
        # 有些行业个股权重以前的不够了
        x = cp.Variable(len(ret_tmp2), nonneg=True)
        y = cp.Variable(len(ret_tmp2), boolean=True)

        para_dict2 = {'x': x,
                      'y': y,
                      'y_sum': max_num,       # - n2,
                      'max_wei': max_wei2,
                      'min_wei': min_wei2,
                      'in_benchmark_wei': in_benchmark_wei,
                      'is_in_bench': is_in_bench2,
                      'ret_e': ret_tmp2,
                      'dum': dum2,
                      'wei_tmp': wei_tmp2,
                      'ind_wei': ind_wei2,    # ind_wei2.sum()
                      'risk_factor_dict': {},  # control_factor_dict,
                      'limit_factor_df': limit_factor_df2,
                      'pre_w': pre_w,
                      'P': P2,
                      'total_wei': tw,
                      }
        con_dict2 = {'in_benchmark': in_benchmark,
                     'industry_expose_control': False,  # industry_expose_control,
                     'industry_max_expose': industry_max_expose,
                     'turnover': turnover,
                     'te': te,
                     }
        q2 = ret_tmp2.values
        cons2 = generates_constraints(para_dict2, con_dict2)
        prob = cp.Problem(cp.Maximize(q2.T * x - cp.quad_form(x, P2)), cons2)
        prob.solve(solver=cp.ECOS_BB, feastol=1e-10)
        print(prob.status)
        status = prob.status
        if status != 'optimal' or status == 'optimal_inaccurate':
            iters = 0
            while status != 'optimal' and iters < 4:
                if iters == 0:
                    if len(control_factor_dict) > 0:
                        tmp_d = deepcopy(control_factor_dict)
                        for k, v in tmp_d.items():
                            tmp_d[k] = v + 2
                        para_dict['risk_factor_dict'] = tmp_d

                if iters == 1:
                    if turnover:
                        turnover = turnover + 0.2
                        con_dict['turnover'] = turnover
                    else:
                        para_dict['risk_factor_dict'] = {}

                if iters == 2:
                    industry_max_expose = industry_max_expose + 0.05
                    con_dict['industry_max_expose'] = industry_max_expose

                iters = iters + 1
                cons2 = generates_constraints(para_dict2, con_dict2)
                prob = generates_problem(q2, x, lam, P2, c, pre_w, cons2, te)
                print('第{}次优化'.format(iters))
                try:
                    prob.solve(solver=cp.ECOS, verbose=False)
                    status = prob.status
                except Exception as e:
                    print('Solver ECOS failed')

        # 如果解不出最优解就使用暴力删除法
        if status != 'optimal' or status == 'optimal_inaccurate':
            wei_se = wei_se.where(wei_se > 0.001, 0)
        else:
            wei_ar = np.array(x.value).flatten()  # wei_ar.size
            wei_se = pd.Series(wei_ar, index=ret_tmp.index)

    pre_sum = wei_se.sum()
    wei_se = wei_se.where(wei_se > 0.001, 0)
    if isinstance(sus_wei, pd.Series):
        # 剔除一些权重特别小的股票后的权重再分配
        wei_se = wei_se * (pre_sum/wei_se.sum())
        # 与前期停牌股票的权重合并
        wei_se = pd.concat([wei_se, sus_wei])
    else:
        wei_se = wei_se/wei_se.sum()

    return wei_se


# 每个行业使用
def optimization_fun_v2(ret, industry_wei_dict, max_num, max_wei=0.05):
    data = Data()
    basic = data.stock_basic_inform
    industry_sw = basic[['申万一级行业']]
    # 股票组合的行业虚拟变量
    industry_map = industry_sw.loc[ret.index, :]

    # dummies_bench = pd.get_dummies(industry_map.loc[bench_wei.index, :])
    # dummies_bench.sum()  不同行业的公司数量
    industry_map.fillna('综合', inplace=True)
    dummies = pd.get_dummies(industry_map[industry_map.columns[0]])

    q = ret.values
    x = cp.Variable(len(ret), nonneg=True)

    try:
        if len(dummies.columns) != len(industry_wei_dict.keys()):
            to_del = [i for i in industry_wei_dict.keys() if i not in dummies.columns]
            for d in to_del:
                industry_wei_dict.pop(d)

        wei_max = []
        wei_min = []
        for v in industry_wei_dict.values():
            wei_min.append(v[0])
            wei_max.append(v[1])

        industry_wei_max = pd.Series(index=list(industry_wei_dict.keys()), data=wei_max)
        industry_wei_min = pd.Series(index=list(industry_wei_dict.keys()), data=wei_min)

        constraints = [x <= max_wei]
        constraints.append(x @ dummies <= industry_wei_max)
        constraints.append(x @ dummies >= industry_wei_min)
        constraints.append(cp.sum(x) == 1)

        prob = cp.Problem(cp.Maximize(q.T * x), constraints)
        prob.solve(solver=cp.ECOS_BB, feastol=1e-10)
        print(prob.status)
        status = prob.status

        wei_ar = np.array(x.value).flatten()  # wei_ar.size
        try:
            wei_se = pd.Series(wei_ar, index=ret.index)
        except Exception as e:
            print('deubg')

        wei_se = wei_se.where(wei_se > 0.001, 0)
        wei_se = wei_se/wei_se.sum()

        if (wei_se > 0).sum() > max_num:
            print('debug')

    except Exception as e:
        wei_se = pd.Series(1, index=ret.index)
        wei_se = wei_se / wei_se.sum()

    return wei_se

