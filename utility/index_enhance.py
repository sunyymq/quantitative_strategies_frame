# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:07:47 2019

@author: HP
"""
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from collections import defaultdict
from seaborn import heatmap
from copy import deepcopy
from scipy.optimize import linprog as lp
from utility.single_factor_test import Backtest_stock
from utility.factor_data_preprocess import info_cols
from utility.tool0 import Data
from utility.constant import industry_benchmark
from utility.single_factor_test import (get_factor_names, panel_to_matrix,
                                        regress)


plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['figure.figsize'] = (16.0, 9.0)  # 图片尺寸设定（宽 * 高 cm^2)
plt.rcParams['font.size'] = 15                # 字体大小

from utility.constant import sf_test_save_path, work_dir, info_cols, total_result_path
from utility.constant import factor_path as factor_panel_path
from utility.constant import factor_matrix_path, rm_save_path, index_enhance_dir

# industry_benchmark = 'sw'      #中信一级行业
#
# # 自动生成合成、正交因子存放目录
# if not os.path.exists(rm_save_path):
#     os.mkdir(rm_save_path)
#
# # 自动生成指数增强模型结果存放目录
# if not os.path.exists(index_enhance_dir):
#     os.mkdir(index_enhance_dir)


def get_stock_wt_in_index(index):
    """
    获取指数（000300.SH或000905.SH）中各截面期成分股所占权重
    """
    global work_dir
    if index.startswith('000300') or index.startswith('399300') or index.startswith('300') or index.endswith('300'):
        openname = 'hs300_wt.csv'
    elif index.startswith('000905') or index.startswith('399905') or index.startswith('500') or index.endswith('500'):
        openname = 'zz500_wt.csv'
    else:
        msg = f'暂不支持当前指数：{index}'
        raise Exception(msg)

    path = r'D:\pythoncode\IndexEnhancement\指数相关'

    index_wt = pd.read_csv(os.path.join(path, openname),
                           index_col=[0], engine='python')
    index_wt.columns = pd.to_datetime(index_wt.columns)
    return index_wt


# 得到指数的行业权重
def get_indus_wt_in_index(index, indus_level='second'):
    index_wt = get_stock_wt_in_index(index)
    data = Data()
    if indus_level == 'first':
        industry = data.firstindustryname
    elif indus_level == 'second':
        industry = data.secondindustryname

    industry = data.reindex(industry)
    industry = industry.loc[index_wt.index, index_wt.columns]

    indus_wt = pd.DataFrame()
    for d in index_wt.columns:
        # d = index_wt.columns[0]
        tmp_df = pd.concat([index_wt.loc[:, d], industry.loc[:, d]], axis=1)
        tmp_df = tmp_df.dropna()
        tmp_df.columns = ['wt', 'industry']
        indus_wt_tmp = tmp_df['wt'].groupby(tmp_df['industry']).sum()
        indus_wt_df = pd.DataFrame(indus_wt_tmp.values, index=indus_wt_tmp.index, columns=[d])

        indus_wt = pd.concat([indus_wt, indus_wt_df], axis=1)

    indus_wt = indus_wt.fillna(0)
    return indus_wt


def get_factor_corr(factors=None, codes=None, basic_path=None):
    """
    计算因子相关系数
    """
    global factor_path
    if factors is None:
        factors = get_factor_names()
    factors_matrix_dat = get_factor(factors, basic_path=basic_path)
    factors_panel_dat = concat_factors_panel(factors, factors_matrix_dat, 
                                             codes, False, False)
    corrs = []
    for date in sorted(factors_panel_dat.keys()):
        factor_panel = factors_panel_dat[date]
        corrs.append(factor_panel.corr())
        
    avg_corr = reduce(lambda df1, df2: df1 + df2, corrs) / len(corrs)
    return avg_corr


def plot_corr_heatmap(corr, save_path, save_name):
    """
    绘制相关系数热力图
    """
    corrfig_path = os.path.join(save_path, '相关系数图')
    
    if not os.path.exists(corrfig_path):
        os.mkdir(corrfig_path)

    final_path = os.path.join(corrfig_path, f'{save_name}.png')
            
    fig, ax = plt.subplots(1, 1)
    heatmap(corr, 
            linewidths=0.05, 
            vmin=-1, 
            vmax=1,
            annot=True,
            cmap='rainbow')
    fig.savefig(final_path)
    plt.close()


def factor_concat(factors_to_concat, new_factor_name, weight=None):
    """
    因子合成：
    输入：待合并因子的名称(,分隔); 合成后的因子存储名称（自动添加_con后缀）;
          合成权重（默认等权）
    输出：合成后因子的因子截面数据和矩阵数据
    """
    global factor_panel_path, rm_save_path, info_cols
    # 新合成因子名称自动添加 con
    if not new_factor_name.endswith('con'):
        new_factor_name += '_con'
    # 存储地址
    cfactor_spath = os.path.join(rm_save_path, '新合成因子')
    cpanel_spath = os.path.join(cfactor_spath, '因子截面')
    cmatrix_spath = os.path.join(cfactor_spath, '因子矩阵')
    if not os.path.exists(cfactor_spath):
        os.mkdir(cfactor_spath)
        os.mkdir(cpanel_spath)
        os.mkdir(cmatrix_spath)
    
    if ',' in factors_to_concat:
        factors_to_concat = factors_to_concat.split(',')
    
    if weight is None:
        apply_func = np.mean
        col_name = new_factor_name + '_equal'
    else:
        apply_func = lambda df: np.sum(weight*df)
        col_name = new_factor_name
    
    if os.path.exists(os.path.join(cmatrix_spath, col_name+'.csv')):
        print(f'{col_name}因子数据已存在')
        return

    panelfactors = os.listdir(cpanel_spath)
    
    for f in os.listdir(factor_panel_path):
        # 依次打开每个月度数据
        dat = pd.read_csv(os.path.join(factor_panel_path, f), encoding='gbk',
                          engine='python', index_col=[0])
        # 选择对应的目标因子
        factor_dat = dat[factors_to_concat]
        # 使用相应的合成方式合成因子
        factor_concated = factor_dat.apply(apply_func, axis=1)
        # 改列名
        factor_concated.name = col_name
        if panelfactors:
            # 如果存在相应的月度数据，则把这个新的合成因子做为一个新列加入到截面数据中
            panel_dat = pd.read_csv(os.path.join(cpanel_spath, f), encoding='gbk',
                                    engine='python', index_col=[0])
            if col_name in panel_dat.columns:
                del panel_dat[col_name]
            panel_dat = pd.concat([panel_dat, factor_concated], axis=1)
        else:
            # 如果不存在，则新建一个新的
            panel_dat = pd.concat([dat[info_cols], factor_concated], axis=1)
        # 存储截面数据
        panel_dat.to_csv(os.path.join(cpanel_spath, f), encoding='gbk')

    # 把截面数据转为因子矩阵
    panel_to_matrix([col_name], factor_path=cpanel_spath,
                    save_path=cmatrix_spath)
    print(f"创建{col_name}因子数据成功.")


def orthogonalize(factors_y, factors_x, codes=None, index_wt=None):
    """
    因子正交：
    输入：因变量(y)、自变量(x)因子名称（,分隔），类型：字符串
    输出：经过正交的因子截面数据和因子矩阵数据
    """
    global rm_save_path, factor_panel_path, info_cols
    ofactor_spath = os.path.join(rm_save_path, '正交后因子')
    opanel_spath = os.path.join(ofactor_spath, '因子截面')
    omatrix_spath = os.path.join(ofactor_spath, '因子矩阵')
    if not os.path.exists(ofactor_spath):
        os.mkdir(ofactor_spath)
        os.mkdir(opanel_spath)
        os.mkdir(omatrix_spath)
    
    for fac in factors_y:
        if os.path.exists(os.path.join(omatrix_spath, fac+'_ortho.csv')):
            print(f'{fac}_ortho因子数据已存在')
            factors_y.remove(fac)
            
    if len(factors_y) == 0:
        return 

    # todo ???????
    panel_y = concat_factors_panel(factors_y, codes=codes, ind=False, mktcap=False)
    panel_x = concat_factors_panel(factors_x, codes=codes, ind=False, mktcap=False)
       
    ortho_y = {}
    for date in sorted(panel_x.keys()):
        y = panel_y[date]
        X = panel_x[date]
        cur_index_wt = index_wt[date].dropna()

        # 先合并确定当前的基准的codes,删除不在基准范围内的股票。
        data_to_regress = pd.concat([X, y], axis=1)
        mut_index = data_to_regress.index.intersection(cur_index_wt.index)
        data_to_regress = data_to_regress.loc[mut_index, :]
        data_to_regress = data_to_regress.dropna(how='any', axis=0)
        cut_loc = len(y.columns)
        X, ys = data_to_regress.iloc[:, :-cut_loc], data_to_regress.iloc[:, -cut_loc:]
       
        resids = pd.DataFrame()
        params_a = pd.DataFrame()
        for fac in ys.columns:
            y = ys[fac]
            _, params, resid_y = regress(y, X, intercept=True)
            params_a = pd.concat([params_a, params], axis=1)
            resid_y.name = fac + '_ortho'
            resids = pd.concat([resids, resid_y], axis=1)
        ortho_y[date] = resids

    # 存储
    for date in ortho_y.keys():
        cur_panel_ortho = ortho_y[date]
        date_str = str(date)[:10]
        basic_info = pd.read_csv(os.path.join(factor_panel_path, date_str+'.csv'),
                                 encoding='gbk', engine='python', index_col=['No'])
        tmp_info_cols = [col for col in info_cols if col in basic_info.columns]
        basic_info = basic_info[tmp_info_cols]

        new_panel = pd.merge(basic_info, cur_panel_ortho,
                             left_on='code', right_index=True)
        new_panel.to_csv(os.path.join(opanel_spath, date_str+'.csv'), encoding='gbk')
    
    factors_ortho = [fac+'_ortho' for fac in factors_y]
    
    panel_to_matrix(factors_ortho, factor_path=opanel_spath,
                    save_path=omatrix_spath)
    print(f"创建{','.join(factors_ortho)}因子数据成功.")


def get_panel_data(names, fpath, codes):
    res = defaultdict(pd.DataFrame)  
    if not isinstance(names, list):
        names = [names]
    for file in os.listdir(fpath):
        date = pd.to_datetime(file.split('.')[0])
        datdf = pd.read_csv(os.path.join(fpath, file), 
                            encoding='gbk', engine='python', 
                            index_col=['code'])
        for name in names:
            dat = datdf.loc[:, name]
            dat.name = date
            if codes is not None:
                dat = dat.loc[codes]
            res[name] = pd.concat([res[name], dat], axis=1)
    return res


def get_matrix_data(name, fpath, codes=None):
    data = pd.read_csv(os.path.join(fpath, name+'.csv'), 
                       encoding='gbk', engine='python', index_col=[0])
    data.columns = pd.to_datetime(data.columns)
    if codes is not None:
        data = data.loc[codes, :]
    return {name: data}


def get_factor(factor_names, codes=None, plate_name=None, basic_path=None):
    if basic_path:
        factor_paths = [(f, basic_path) for f in factor_names]
    else:
        factor_paths = [(f, get_factor_path(f, plate_name=plate_name)) for f in factor_names]

    factors_matrix = {fname: path for fname, path in factor_paths
                      if path.endswith('因子矩阵')}
    
    factors_panel = defaultdict(list)
    for fname, path in factor_paths:
        if path.endswith('截面') or '预处理' in path:
            factors_panel[path].append(fname)
    
    res = {}
    for fname, fpath in factors_matrix.items():
        res.update(get_matrix_data(fname, fpath, codes))
    
    for fpath, fnames in factors_panel.items():
        res.update(get_panel_data(fnames, fpath, codes))
    return res 


def get_factor_path(factor_name, plate_name=None):
    """
    共三类因子，分别是处理后的股票因子、经过合成正交化后的大类因子、可能的行业特色因子
    """

    path_matrix1 = r'D:\pythoncode\IndexEnhancement\多因子选股\新合成因子\因子矩阵'
    path_matrix2 = r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵'
    path_matrix3 = r'D:\pythoncode\IndexEnhancement\多因子选股\正交后因子\因子矩阵'

    # path_panel1 = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）'
    # path_panel2 = r'D:\pythoncode\IndexEnhancement\多因子选股\正交后因子\因子截面'

    list_p1 = os.listdir(path_matrix1)
    list_p1_n = [fn.split('.')[0] for fn in list_p1]
    list_p2 = os.listdir(path_matrix2)
    list_p2_n = [fn.split('.')[0] for fn in list_p2]
    list_p3 = os.listdir(path_matrix3)
    list_p3_n = [fn.split('.')[0] for fn in list_p3]

    if factor_name in list_p1_n:
        open_path = path_matrix1
    elif factor_name in list_p2_n:
        open_path = path_matrix2
    elif factor_name in list_p3_n:
        open_path = path_matrix3
    else:
        raise TypeError(f"不支持的因子数据格式：{factor_name}")

    return open_path


def concat_factors_panel(factors=None, factors_dict=None, codes=None,
                         ind=True, mktcap=True, perchg_nm=False, basic_path=None):
    '''
    把因子矩阵形式的存储，变成字典形式的存储，每个key是日期，value是行为codes，列为factors的dataframe
    '''
    global factor_panel_path, industry_benchmark
    factors = deepcopy(factors)
    if factors:
        if isinstance(factors, str):
            factors = factors.split(',')
    else:
        factors = []
        
    if ind:
        factors.append(f'Industry_sw')
    if mktcap:
        factors.append('Mkt_cap_float')
    if perchg_nm:
        factors.append('Pct_chg_nm')
    
    if codes is not None and factors_dict is not None:
        factors_dict = {fac: datdf.loc[codes, :] for fac, datdf in factors_dict.items()}
    # 获取待回归因子自变量矩阵数据
    if (factors_dict is None) or ('MKT_CAP_FLOAT' in factors) or ('MKT_CAP_FLOAT' in factors) or\
        (f'industry_{industry_benchmark}' in factors):
        # matrix是一个dict
        matrix = {}
        for fac in factors:
            if basic_path:
                fpath = basic_path
            else:
                fpath = get_factor_path(fac)

            matrix.update(get_matrix_data(fac, fpath, codes))
        if factors_dict:
            matrix.update(factors_dict)
    else:
        matrix = factors_dict
    panel = defaultdict(pd.DataFrame)
    
    # 对每个时间截面，合并因子数据
    facs = sorted(matrix.keys())
    for fac in facs:
        for date in matrix[fac]:
            cur_fac_panel_data = matrix[fac][date]
            cur_fac_panel_data.name = fac
            if 'industry' in fac and (ind == True):
                cur_fac_panel_data = pd.get_dummies(cur_fac_panel_data)
            if fac == 'MKT_CAP_FLOAT' and (mktcap == True):
                cur_fac_panel_data = np.log(cur_fac_panel_data)
                cur_fac_panel_data.name = 'ln_mkt_cap'

            panel[date] = pd.concat([panel[date], cur_fac_panel_data], axis=1)
        
    return panel


def get_exponential_weights(window=12, half_life=6):
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] 


def wt_sum(series, wt):
    if len(series) < len(wt):
        return np.sum(series * wt[:len(series)] / np.sum(wt[:len(series)]))
    else:
        return np.sum(series * wt / np.sum(wt))


def factor_return_forecast(factors_x, factor_data=None, 
                           window=12, half_life=6, only_in_index=False):
    """
    因子收益预测：
    输入：自变量(x)因子名称（,分隔），类型：字符串
    输出：截面回归得到的因子收益率预测值，行：因子名称，列：截面回归当期日期
    only_in_index 表示是否只对指数成分股进行预测，True表示只对指数成分股进行预测，False表示对全部股票进行预测
    """
    global factor_panel_path

    if only_in_index:
        index_wt = get_stock_wt_in_index('000300.SH')
    ret_matrix = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']
        
    if factor_data is None:
        panel_x = concat_factors_panel(factors_x)
    else:
        panel_x = factor_data
    
    # 截面回归，获取回归系数，作为因子收益，factor_rets的index里面包含alpha因子和行业因子
    factor_rets = pd.DataFrame()
    for date in sorted(panel_x.keys()):
        y = ret_matrix[date]
        X = panel_x[date]

        # 先合并
        data_to_regress = pd.concat([X, y], axis=1)
        # 删除nan
        data_to_regress = data_to_regress.dropna(how='any', axis=0)

        if only_in_index:
            cur_index_wt = index_wt[date].dropna()
            # 找到在当期指数里的标的
            mut_index = data_to_regress.index.intersection(cur_index_wt.index)
            # 选择仅在指数里的标的
            data_to_regress = data_to_regress.loc[mut_index, :]

        # 分成X,y
        X, y = data_to_regress.iloc[:, :-1], data_to_regress.iloc[:, -1]
        # 有些行业可能不在，删除相应的行业哑变量
        for fac in X.sum()[X.sum() == 0].index:
            if fac not in factors_x:
                del X[fac]
        w = X['ln_mkt_cap']
        # 回归并存储结果
        _, cur_factor_ret, _ = regress(y, X, w)
        cur_factor_ret.name = date
        factor_rets = pd.concat([factor_rets, cur_factor_ret], axis=1)
    
    # 对ROE_q以及growth因子的负值纠正为0
    factors_to_correct = ['ROE_q', 'growth']
    factor_rets = factor_rets.T
    for fac in factors_to_correct:
        try:
            fac_name = [f for f in factor_rets.columns if f.startswith(fac)][0]
        except IndexError:
            continue
        factor_rets[fac_name] = factor_rets[fac_name].where(factor_rets[fac_name] >= 0, 0)
    
    # 指数加权权重-窗口平滑处理
    if half_life:
        exp_wt = get_exponential_weights(window=window, 
                                         half_life=half_life)
        factor_rets = factor_rets.rolling(window=window, min_periods=1).\
                apply(wt_sum, args=(exp_wt,)).shift(1)
    else:
        factor_rets = factor_rets.rolling(window=window, min_periods=1).\
                mean().shift(1)
    factor_rets = factor_rets.dropna(how='all', axis=0)
    return factor_rets


def get_est_stock_return(factors, factors_panel, est_factor_rets, 
                         window=12, half_life=6):
    """
    根据截面回归所得各期系数（因子收益率）, 得到各股票的截面预期收益
    """
    est_stock_rets = pd.DataFrame()
    for date in est_factor_rets.index:
        # date = est_factor_rets.index[-2]
        cur_factor_panel = factors_panel[date]
        try:
            cur_factor_panel = cur_factor_panel[factors]
        except Exception as e:
            print('look')
        cur_factor_panel = cur_factor_panel.dropna(how='any', axis=0)
        cur_est_stock_rets = np.dot(cur_factor_panel, 
                                    est_factor_rets.loc[date, factors])
        cur_est_stock_rets = pd.DataFrame(cur_est_stock_rets, 
                                          index=cur_factor_panel.index, 
                                          columns=[date])
        est_stock_rets = pd.concat([est_stock_rets, cur_est_stock_rets], 
                                   axis=1)

    return est_stock_rets


def get_refresh_days(tradedays, start_date, end_date):
    """
    获取调仓日期（回测期内的每个月首个交易日）
    """
    tdays = tradedays
    sindex = get_date_idx(tradedays, start_date)
    eindex = get_date_idx(tradedays, end_date)
    tdays = tdays[sindex:eindex+1]
    return (nd for td, nd in zip(tdays[:-1], tdays[1:]) 
            if td.month != nd.month)


def get_date_idx(tradedays, date):
    """
    返回传入的交易日对应在全部交易日列表中的下标索引
    """
    datelist = list(tradedays)
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


def plot_net_value(records, benchmark, method_name, save_path, num):
    """
    绘制回测净值曲线
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    records['benchmark_nv'] = (records['benchmark'] + 1).cumprod()
    records = records[['benchmark_nv', 'net_value']]
    records /= records.iloc[0,:]
    plt.plot(records)
    plt.legend([benchmark, method_name], loc=2)
    plt.title('回测净值')
    plt.savefig(os.path.join(save_path, f'{method_name}_{num}.png'))
    plt.close()


def lp_solve(date, cur_est_rets, limit_factors, cur_benchmark_wt, industry_map, stocks_in_index_wei=1, num_multi=5):
    """
    线性规划计算函数：
    输入：截面预期收益，约束条件（风险因子），截面标的指数成分股权重，个股权重约束倍数
    输出：经优化后的组合内个股权重
    """

    '''
    scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='interior-point',
                           callback=None, options=None, x0=None)
    minimize:
        c @ x
    such that:
        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub
    '''
    # b_eq是基准的风险暴露，通过是组合的风险暴露与基准的风险暴露相同，达到风险因子中性、行业中性的目的
    # np.r_ 是按列拼接成一个maxic，就是在风险因子权重等于指数风险因子权重，且权重和为1。

    # 行业虚拟变量
    dummies = pd.get_dummies(industry_map[industry_map.columns[0]])
    # 合并
    data = pd.concat([cur_est_rets, limit_factors, cur_benchmark_wt, dummies], axis=1)

    # 仅在成份股中选股
    if stocks_in_index_wei == 1:
        # 把不在成份股的都删掉了
        data = data.dropna(how='any', axis=0)
        cur_est_rets, limit_factors, cur_benchmark_wt, dummies = (data[cur_est_rets.columns], data[limit_factors.columns],
                                                                  data[cur_benchmark_wt.columns], data[dummies.columns])
        cur_benchmark_wt = cur_benchmark_wt / cur_benchmark_wt.sum()

        c = cur_est_rets.values.flatten()
        A_ub = None
        b_ub = None
        A_eq = np.r_[limit_factors.T.values, dummies.T.values, np.repeat(1, len(limit_factors)).reshape(1, -1)]
        b_eq = np.r_[np.dot(limit_factors.T, cur_benchmark_wt), np.dot(dummies.T, cur_benchmark_wt),
                     np.array([1]).reshape(-1, 1)]

        np.repeat(1, 6).reshape(1, -1)

        # 股票权重的bounds,最小是0，最大是指数权重的5倍。
        bounds = tuple([(0, num_multi * wt_in_index) for wt_in_index in cur_benchmark_wt.values])
        try:
            res = lp(-c, A_ub, b_ub, A_eq, b_eq, bounds)
        except Exception as e:
            res = lp(-c, A_ub, b_ub, A_eq, b_eq, bounds, method='interior-point')
            print('{}简单方法未解出来'.format(date))
        cur_wt = pd.Series(res.x, index=cur_est_rets.index)

        # (cur_wt > 0).sum()
        # (cur_wt > 0.01).sum()

    # 可以选出部分的非成份股
    else:
        # 把不在成份股的股票权重nan部分变为0，这样后面就不会被删除
        data[cur_benchmark_wt.columns] = data[cur_benchmark_wt.columns].fillna(0)
        # 删除其他的nan
        data = data.dropna(how='any', axis=0)
        cur_est_rets, limit_factors, cur_benchmark_wt, dummies = (data[cur_est_rets.columns], data[limit_factors.columns],
                                                                  data[cur_benchmark_wt.columns], data[dummies.columns])

        cur_benchmark_wt = cur_benchmark_wt / cur_benchmark_wt.sum()
        # 判断是否为成份股
        not_in_benchmark = deepcopy(cur_benchmark_wt)
        not_in_benchmark[cur_benchmark_wt == 0.0] = 1
        not_in_benchmark[cur_benchmark_wt != 0.0] = 0

        c = cur_est_rets.values.flatten()
        A_ub = not_in_benchmark.T.values
        b_ub = np.array([1 - stocks_in_index_wei])

        A_eq = np.r_[limit_factors.T.values, dummies.T.values, np.repeat(1, len(limit_factors)).reshape(1, -1)]
        b_eq = np.r_[np.dot(limit_factors.T, cur_benchmark_wt), np.dot(dummies.T, cur_benchmark_wt),
                     np.array([1]).reshape(-1, 1)]

        # 得到行业权重
        tmp = pd.concat([cur_benchmark_wt, industry_map], axis=1).fillna(0)
        grouped = tmp.groupby(tmp.columns[-1])
        tmp_indus_wei = pd.Series()
        for k, v in grouped:
            su = v[cur_benchmark_wt.columns].sum().values[0]
            tmp_indus_wei[k] = su
        tmp['indus_wei'] = None
        for i in range(0, len(tmp.index)):
            tmp.loc[tmp.index[i], 'indus_wei'] = tmp_indus_wei[tmp.loc[tmp.index[i], '申万一级行业']]
        bounds_tmp = []
        for v in tmp['indus_wei'].values:
            if v > 0:
                bounds_tmp.append([(0,  v/3)])
            else:
                bounds_tmp.append([(0, 0.0001)])
        bounds1 = tuple(bounds_tmp)

        bounds = tuple([(0, 1) for i in cur_benchmark_wt.values])

        res = lp(-c, A_ub, b_ub, A_eq, b_eq, bounds, method='interior-point')
        res.x
        cur_wt11 = pd.Series(res.x, index=cur_est_rets.index)
        (cur_wt11 > 0).sum()
        (cur_wt11 > 0.001).sum()
        (cur_wt11 > 0.03).sum()

        cur_wt11.sum()

        res = lp(-c, A_ub, b_ub, A_eq, b_eq, bounds)
        cur_wt = pd.Series(res.x, index=cur_est_rets.index)

    return cur_wt


def linear_programming(data_dict, industry_neutralized=False, mv_neutralized=False, equal_weighted=False):
    """
    线性规划法-求解最优组合权重
    """
    est_stock_rets, limit_fac_data, index_wt = data_dict['est_stock_rets'], \
                        data_dict['limit_fac_data'], data_dict['index_wt']
    stock_wt = pd.DataFrame()
    data = Data()
    basic = data.stock_basic_inform
    industry_sw = basic[['申万一级行业']]
    for date in est_stock_rets.columns:
        est_rets = est_stock_rets[[date]].dropna()
        est_rets.columns = ['rets']
        limit_fac_panel = limit_fac_data[date].dropna()
        benchmark_wt = index_wt[[date]].dropna()
        benchmark_wt.columns = ['benchmark_wt']

        cur_wt = lp_solve(date, est_rets, limit_fac_panel, benchmark_wt, industry_sw)
        cur_wt.name = date
        stock_wt = pd.concat([stock_wt, cur_wt], axis=1)
    
    stock_wt = stock_wt.where(stock_wt != 0, np.nan)
    return stock_wt        


def stratified_sample(data_dict):
    """
    分层抽样法-求解组合最优权重
    """
    data_panel = concat_factors_panel(None, data_dict, None, False, False)
    
    stock_wt = pd.DataFrame()
    for date in sorted(data_panel.keys()):
        panel = data_panel[date]
        if 'est_stock_rets' not in panel.columns:
            continue
        panel = panel.dropna(how='any', axis=0)
        panel_stkwt = pd.Series()
        for name, df in panel.groupby('industry_zx'):
            num = len(df) // 3
            remainder = len(df) % 3
            if len(df) <= 3:
                cur_ind_wt = df['index_wt']
                panel_stkwt = pd.concat([panel_stkwt, cur_ind_wt], axis=0)
            else:
                df = df.sort_values(by='MKT_CAP_FLOAT', ascending=False)
                if remainder == 1:
                    cut1, cut2 = num + 1, 2 * num + 1
                elif remainder == 2:
                    cut1, cut2 = num + 1, 2 * num + 2
                else:
                    cut1, cut2 = num, 2 * num
                df1, df2, df3 = df.iloc[:cut1, :], \
                            df.iloc[cut1:cut2, :], df.iloc[cut2:,:]
                for mkt_cap_group in [df1, df2, df3]:
                    max_code_idx = np.argmax(mkt_cap_group['est_stock_rets'])
                    cur_ind_wt = mkt_cap_group.loc[[max_code_idx], 'index_wt']
                    cur_ind_wt.loc[:] = np.sum(mkt_cap_group['index_wt'])
                    panel_stkwt = pd.concat([panel_stkwt, cur_ind_wt], axis=0)
        panel_stkwt.name = date
        stock_wt = pd.concat([stock_wt, panel_stkwt], axis=1)
    
    return stock_wt


def performance_attribution(factors_dict, index_wt, stock_wt, start_date, end_date):
    """
    业绩归因
    """

    factors_panel = concat_factors_panel(None, factors_dict, None, False, False)
    dates = stock_wt.loc[:, start_date:end_date].columns
    dates = pd.to_datetime(dates)
    res = pd.DataFrame()
    for date in dates:
        cur_index_wt = index_wt[date] / 100
        cur_index_wt = cur_index_wt / cur_index_wt.sum()
        w_delta = stock_wt[date] - cur_index_wt
        w_delta = w_delta.dropna()
        cur_factors_panel = factors_panel[date].loc[w_delta.index, :]
        cur_factor_exposure = w_delta.T @ cur_factors_panel
        cur_factor_exposure.name = date
        res = pd.concat([res, cur_factor_exposure], axis=1)
        
    res = res.T.groupby(pd.Grouper(freq='y')).mean()
    return res


def get_market_data(type='second_industry'):
    if type == 'stock':
        market_data = pd.read_csv(os.path.join(work_dir, 'pct_chg.csv'),
                                  engine='python', index_col=[0])
        market_data.columns = pd.to_datetime(market_data.columns)
    elif type == 'second_industry':
        path = f'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\单因子检验\因子矩阵\PCT_CHG_NM.csv'
        market_data = pd.read_csv(path, engine='python', index_col=[0])
        market_data.columns = pd.to_datetime(market_data.columns)

    return market_data


def get_ori_name(factor_name, factors_to_concat):
    if 'ortho' in factor_name:
        factor_name = factor_name[:-6]
    if 'con' in factor_name:
        pat = re.compile('(.*)_con_')
        ori_name = re.findall(pat, factor_name)[0]
        return factors_to_concat[ori_name]
    else:
        return [factor_name]


def factor_process(method, factors_to_concat, factors_ortho, 
                   index_wt, mut_codes, factors, risk_factors=None):
    '''
    :param method:             分层抽样 or 线性规划
    :param factors_to_concat:  需要合成的因子名称和次级因子名称
    :param factors_ortho:      需要正交的因子
    :param index_wt:           历史上所有的指数权重
    :param mut_codes:          所有进过HS300的股票代码
    :param factors:            alpha因子
    :param risk_factors:       风险因子
    '''

    # 因子处理-线性规划
    #   因子合成（等权）
    for factor_con, factors_to_con in factors_to_concat.items():
        factor_concat(factors_to_con, factor_con)
    #   因子正交
    for factor_x, factors_y in factors_ortho.items():
        orthogonalize(factors_y, factor_x, mut_codes, index_wt)

    # 若有 risk_factors，就把factor和risk_factors相加，若没有，就近是factors
    factors_to_corr = factors + risk_factors if risk_factors else factors
    factors_to_corr_ori = [name for fac in factors_to_corr
                           for name in get_ori_name(fac, factors_to_concat)]
    corr_ori = get_factor_corr(factors_to_corr_ori, mut_codes)
    plot_corr_heatmap(corr_ori, method, preprocessed=False)
    corr = get_factor_corr(factors_to_corr, mut_codes)
    plot_corr_heatmap(corr, method, preprocessed=True)
    print("相关系数热力图绘制完毕...")


def index_enhance_model(method='l', benchmark='000300.SH', 
                        start_date=None, end_date=None, methods=None):  
    global index_enhance_dir
    lp_save_path = os.path.join(index_enhance_dir, '线性规划')
    ss_save_path = os.path.join(index_enhance_dir, '分层抽样')

    market_data = get_market_data(type='stock')
    pctchgnm = get_factor(['PCT_CHG_NM'])['PCT_CHG_NM']
    index_wt = get_stock_wt_in_index(benchmark)    
    mut_codes = index_wt.index.intersection(pctchgnm.index)

    if method == 'l':
        method_name = 'linear_programming'
        save_path = lp_save_path
    elif method == 's':
        method_name = 'stratified_sample'
        save_path = ss_save_path
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    params = methods[method_name]
    factors, window, half_life = params['factors'], params['window'], \
                        params['half_life']
    data_dict = {}
    if method == 'l':
        risk_factors = params['risk_factors']
        factors.extend(risk_factors)
    elif method == 's':
        factors.extend(['industry_zx', 'MKT_CAP_FLOAT'])

    # 得到所有因子的因子矩阵，以dict形式存储
    factors_dict = {fac: get_factor([fac], mut_codes)[fac] for fac in factors}
    
    if method == 'l':
        # 得到风险因子的因子矩阵，以dict形式存储，key为因子名称。
        risk_fac_data = {k: v for k, v in factors_dict.items() if k in risk_factors}
        # 把因子矩阵形式的存储，变成字典形式的存储，每个key是日期，value是行为codes，列为factors的dataframe
        limit_fac_data = concat_factors_panel(risk_factors, risk_fac_data, mut_codes,
                                              ind=True, mktcap=False)
        # 存进变量
        data_dict.update({'limit_fac_data': limit_fac_data})
        # 剔除风险因子
        for fac in risk_factors:
            factors_dict.pop(fac)
            factors.remove(fac)
    elif method == 's':
        ind_mktcap_data = {k: v for k, v in factors_dict.items() if k in ['industry_zx', 'MKT_CAP_FLOAT']}
        data_dict.update(ind_mktcap_data)
        for fac in ['industry_zx', 'MKT_CAP_FLOAT']:
            factors_dict.pop(fac)
            factors.remove(fac)
            
    # 将alpha因子整理为截面形式
    factors_panel = concat_factors_panel(None, factors_dict, mut_codes)
    # 因子收益预测模型
    est_fac_rets = factor_return_forecast(factors, factors_panel, window, 
                                          half_life)
    est_fac_rets = est_fac_rets[factors]

    # 全部股票收益预测
    est_stock_rets = get_est_stock_return(factors, factors_panel,
                                          est_fac_rets, window, half_life)
    print('计算股票预期收益率完成...')    

    # 选择出在指数内股票的预测收益
    mut_dates = index_wt.columns.intersection(est_stock_rets.columns) 
    index_wt = index_wt.loc[mut_codes, mut_dates]  
    est_stock_rets = est_stock_rets.loc[mut_codes, mut_dates]
    est_stock_rets.name = 'est_stock_return'

    data_dict.update({'index_wt': index_wt, 
                      'est_stock_rets': est_stock_rets})
    # 优化股票权重
    wt_cal_func = globals()[method_name]
    stock_wt = wt_cal_func(data_dict)
    stock_wt = stock_wt / stock_wt.sum()
    print('计算股票权重完成...')
    
    all_codes = stock_wt.index
    benchmarkdata = market_data.loc[benchmark, start_date:end_date].T
    market_data = market_data.loc[all_codes, start_date:end_date]
    
    # 根据优化得到的各月末截面期HS300成分股股票权重，进行回测
    bt = Backtest_stock(market_data=market_data, 
                        start_date=start_date, 
                        end_date=end_date,
                        benchmarkdata=benchmarkdata, 
                        stock_weights=stock_wt,
                        use_pctchg=True)
    bt.run_backtest()
    print('回测结束, 进行回测结果分析...')
    summary_yearly = bt.summary_yearly()  # 回测统计
    # 业绩归因
    p_attr = performance_attribution(factors_dict, index_wt, stock_wt, 
                                     start_date, end_date)
    
    if os.listdir(save_path):
        try:
            ori_max_num = sorted(f.split('.')[0] for f in os.listdir(save_path) 
                             if f.endswith('csv'))[-1].split('_')[-1]
            cur_num = 1 + int(ori_max_num)
        except:
            cur_num = 1
    else:
        cur_num = 1
    
    # 存储回测结果
    summary_yearly.to_csv(os.path.join(save_path, f'回测统计-分年_{cur_num}.csv'), 
                   encoding='gbk')
    bt.portfolio_record.to_csv(os.path.join(save_path, f'回测净值情况_{cur_num}.csv'), 
                               encoding='gbk')
    bt.position_record.to_csv(os.path.join(save_path, f'各期持仓_{cur_num}.csv'), 
                              encoding='gbk')
    p_attr.to_csv(os.path.join(save_path, f'业绩归因_{cur_num}.csv'), 
                  encoding='gbk')
    plot_net_value(bt.portfolio_record, benchmark, method_name, save_path, cur_num)
    print("分析结果存储完成!")



