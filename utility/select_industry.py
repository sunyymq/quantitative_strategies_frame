# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:07:47 2019

@author: HP
"""
import shutil
from datetime import datetime
from utility.index_enhance import *
from utility.constant import copy_cols
from utility.constant import select_indus_dir
from sklearn.covariance import LedoitWolf
from sklearn import preprocessing
import cvxpy as cp
from cvxopt import solvers, matrix
from scipy.optimize import minimize


def apply_func(df, type, wei=None):
    if type == 'equal_weight':
        return df.mean()
    elif type == 'max_ic_ir':
        # series对应元素相差再求和
        value = np.sum(df * wei)
        return value


def my_factor_concat(path_dict, factors_dict, status='update', concat_type='equal_weight',
                     icir_window=6, start_date=None):
    """
    因子合成：
    输入：待合并因子的名称(,分隔); 合成后的因子存储名称（自动添加_con后缀）;
          合成权重（默认等权）
    输出：合成后因子的因子截面数据和矩阵数据

    与原始的 factor_concat的区别，存储时的名字没加_con，没加_equal
    """
    global info_cols

    save_path = path_dict['save_path']
    factor_panel_path = path_dict['factor_panel_path']
    # ic_path = path_dict['ic_path']

    # 存储地址
    cfactor_spath = os.path.join(save_path, '新合成因子')
    cpanel_spath = os.path.join(cfactor_spath, '因子截面')
    cmatrix_spath = os.path.join(cfactor_spath, '因子矩阵')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(cfactor_spath):
        os.mkdir(cfactor_spath)
    if not os.path.exists(cpanel_spath):
        os.mkdir(cpanel_spath)
    if not os.path.exists(cmatrix_spath):
        os.mkdir(cmatrix_spath)

    # if ',' in factors_to_concat:
    #     factors_to_concat = factors_to_concat.split(',')
    #
    # factors_to_concat = [f.replace('_div_', '/') for f in factors_to_concat]
    # col_name = new_factor_name

    # 仅更新新月份的结果
    if status == 'update':
        # 首先从后往前打开文件，看pctchange_nm是否为空，如果为空则删除该文件。不为空就退出该任务。
        # 主要解决最后一期pctchange为空的问题。

        # 选择出需要更新的文件
        hased_list = [pd.Timestamp(datetime.strptime(m.split('.')[0], "%Y-%m-%d")) for m in os.listdir(cpanel_spath)]
        fd_list = [pd.Timestamp(datetime.strptime(m.split('.')[0], "%Y-%m-%d")) for m in os.listdir(factor_panel_path)]

        if len(hased_list) > 0:
            to_compute = [m for m in fd_list if m not in hased_list and m > hased_list[-1]]
        else:
            to_compute = fd_list

        if len(to_compute) == 0:
            print(f"因子数据无需要更新的，退出.")
            return 0
        else:
            if len(hased_list) > 0:
                to_compute = [hased_list[-1]] + to_compute
            dirlist = [m.to_pydatetime().strftime("%Y-%m-%d") + '.csv' for m in to_compute]

    elif status == 'renew':
        dirlist = os.listdir(factor_panel_path)
        # 对于开始日期前的数据就不合并处理了。
        if start_date:
            dirlist = [dir_tmp for dir_tmp in dirlist if datetime.strptime(dir_tmp.split('.')[0], "%Y-%m-%d") >= start_date]

    if status == 'update':
        st = 0
    elif status == 'renew':
        if concat_type == 'max_ic_ir':
            st = icir_window + 1
        elif concat_type == 'equal_weight':
            st = 0

    for f in range(st, len(dirlist)):
        # f = 119
        fin = dirlist[f]
        # 依次打开每个月度数据
        dat = pd.read_csv(os.path.join(factor_panel_path, fin), encoding='gbk',
                          engine='python')
        if 'Code' in dat.columns:
            dat = dat.set_index('Code')
        tmp_cols = [col for col in info_cols if col in dat.columns]
        panel_dat = dat[tmp_cols]

        for col_name, factors_to_concat in factors_dict.items():
            # 选择存在的目标因子
            to_con = [f for f in factors_to_concat if f in dat.columns]
            if len(to_con) == 0:
                continue

            factor_dat = dat[to_con]
            # 使用相应的合成方式合成因子
            if len(to_con) == 1:     # 仅是重命名
                factor_concated = factor_dat
            elif concat_type == 'equal_weight':
                factor_concated = factor_dat.apply(apply_func, args=(concat_type,), axis=1)

            # 改列名
            if isinstance(factor_concated, pd.Series):
                factor_concated = pd.DataFrame(factor_concated)
            factor_concated.columns = [col_name]

            panel_dat = pd.concat([panel_dat, factor_concated], axis=1)

        # 对新和成的因子做Zscore处理
        cols = [k for k in factors_dict.keys() if k in panel_dat.columns]
        try:
            scaled = preprocessing.scale(panel_dat[cols])
        except Exception as e:
            print('debug')
        scaled_df = pd.DataFrame(data=scaled, index=panel_dat.index, columns=cols)
        panel_dat[cols] = scaled_df

        panel_dat = panel_dat.reset_index()
        if 'No' in panel_dat.columns:
            panel_dat = panel_dat.set_index('No')
        # 存储截面数据
        panel_dat.to_csv(os.path.join(cpanel_spath, fin), encoding='gbk')

    print(f"创建因子数据成功.")


# 主要解决miatrix的地址问题，在因子合成，因子正交的时间，在新的matrix地址上，把其他常用的
# matrix一起拷贝过来。
def copy_matrix(old_p, new_p, factors_name=None):
    if not factors_name:
        factors_name = copy_cols
    dirlist = os.listdir(old_p)

    factors_name = [f for f in factors_name if f in dirlist]
    for f in factors_name:
        shutil.copyfile(os.path.join(old_p, f), os.path.join(new_p, f))


def fun(x, mean, f_s):
    '''
    :param wei:  要求的权重，wei是1行N列的array
    :param mean: 历史IC均值，mean是N行1列的array
    :param f_s:  IC协方差矩阵，
    :return:
    '''
    # where x is an 1 - Darray with shape(n, )
    value = np.mat(x) * np.mat(mean) / np.sqrt(np.mat(x) * np.mat(f_s) * np.mat(x).T)
    return value


def weight_optimizer(factors_to_concat, factor_path, window):
    '''
    :param factors_to_concat: 待合成因子的名称
    :param factor_path:  因子月度IC值得存储文件
    :param window:  历史回溯期长度
    :return max_icir_weight 因子权重
    '''

    history_ic = pd.read_csv(factor_path, engine='python')
    history_ic = history_ic.set_index(history_ic.columns[0])
    new_col = [col.lower() for col in history_ic.columns]
    history_ic.columns = new_col
    history_ic = history_ic[factors_to_concat]

    wei_pd = pd.DataFrame(index=history_ic.index, columns=factors_to_concat)
    for i in range(window, len(history_ic)):
        ic_value = history_ic.iloc[i-window:i, :]
        ic_value = ic_value.fillna(0)
        mean = ic_value.mean(axis=0)
        '''
        优化问题为： max ic_ir = wei * ic_mean / 根号下 wei*协方差*wei.t
        s.t.  wei >= 0 
        '''
        cov = LedoitWolf().fit(ic_value.values)
        f_shrinkaged = cov.covariance_
        mean_v = mean.values
        bnds = []
        for j in range(0, len(factors_to_concat)):
            bnds.append((0, 1))
        bnds = tuple(bnds)

        x0 = np.ones(len(factors_to_concat))/len(factors_to_concat)
        args = (mean_v.reshape(len(factors_to_concat), 1), f_shrinkaged)
        # TODO: 要最大值，求得是最小值
        res = minimize(fun, x0, args=args, bounds=bnds)
        wei = res.x
        wei = wei/np.sum(wei)
        wei_pd.loc[wei_pd.index[i], :] = wei
    wei_pd = wei_pd.dropna(how='any')
    wei_pd = wei_pd.shift(periods=1, axis=0)

    return wei_pd


def quadraticprogramming(alpha, e_factor, e_shrinkaged, u=1):
    # 二次规划标准形式：
    #    MIN: (1/2)*x.T*P*x + q.Tx
    # subject to: G*x <= h
    #          Ax = b
    # sol = solvers.qp(P,q,G,h,A,b)
    # x为待求解的列向量
    e_factor = matrix(e_factor)
    e_shrinkaged = matrix(e_shrinkaged)

    P = 2 * u * (e_factor + e_shrinkaged)
    q = matrix(-1 * alpha)
    G = matrix(-1 * np.eye(100))
    h = matrix(-0.002, (100, 1))  # 未来避免过小的值，设置最小权重为0.001
    A = matrix(1.0, (1, 100))
    b = matrix(1.0)
    # print(h)
    sol = solvers.qp(P, q, G, h, A, b)
    # print(sol['x'])
    wei = sol['x']
    return wei


def Symmetry(factors, path_dict):
    '''
    :param factors: 需要做中性化处理的因子
    :return:
    '''
    global info_cols

    save_path = path_dict['save_path']
    # 使用新合成因子的因子截面数据，来做被中性化处理的数据
    factors_path = os.path.join(save_path, '新合成因子', '因子截面')
    # 存储的地址
    after_sym_path = os.path.join(save_path, '正交后因子')
    after_sym_panel_spath = os.path.join(after_sym_path, '因子截面')
    after_sym_matrix_spath = os.path.join(after_sym_path, '因子矩阵')
    if os.path.exists(after_sym_path):
        # 删除历史数据
        shutil.rmtree(after_sym_path)

    if not os.path.exists(after_sym_path):
        os.mkdir(after_sym_path)
    if not os.path.exists(after_sym_panel_spath):
        os.mkdir(after_sym_panel_spath)
    if not os.path.exists(after_sym_matrix_spath):
        os.mkdir(after_sym_matrix_spath)

    for f in os.listdir(factors_path):
        # f = os.listdir(factors_path)[0]
        # 依次打开每个月度数据
        dat = pd.read_csv(os.path.join(factors_path, f), encoding='gbk',
                          engine='python', index_col=[0])
        tmp_cols = [col for col in info_cols if col in dat.columns]
        try:
            factor_to_sym = dat[factors]
        except Exception as e:
            print('debug')
        factor_to_sym_col_name = factor_to_sym.columns
        D, U = np.linalg.eig(np.dot(factor_to_sym.T, factor_to_sym))
        S = np.dot(U, np.diag(D ** (-0.5)))

        Fhat = np.dot(factor_to_sym, S)
        Fhat = np.dot(Fhat, U.T)

        sym_panel = pd.DataFrame(Fhat, columns=factor_to_sym_col_name, index=factor_to_sym.index)
        sym_panel = pd.concat([dat[tmp_cols], sym_panel], axis=1)

        sym_panel.to_csv(os.path.join(after_sym_panel_spath, f), encoding='gbk')

    # 把截面数据转为因子矩阵
    panel_to_matrix(factors, factor_path=after_sym_panel_spath, save_path=after_sym_matrix_spath)


def history_factor_return(factors_name, factors_panel, window=12, half_life=6):
    """
    因子收益预测：
    输入：自变量(x)因子名称（,分隔），类型：字符串
    输出：截面回归得到的因子收益率预测值，行：因子名称，列：截面回归当期日期
    """
    factor_rets = pd.DataFrame()
    resid = pd.DataFrame()
    # 截面回归，获取回归系数，作为因子收益，factor_rets的index里面包含alpha因子和行业因子
    for date in sorted(factors_panel.keys()):
        data_to_regress = factors_panel[date]
        # 删除nan
        data_to_regress = data_to_regress.dropna(how='any', axis=0)
        if data_to_regress.empty:
            # 记录最后一个月的日期，在最后shift的时候添加上
            added_index = date
            continue

        # 分成X,y
        try:
            y = data_to_regress.loc[:, 'Pct_chg_nm']
        except Exception as e:
            print('no pct chag nm')

        x_col = list(data_to_regress.columns)
        x_col.remove('Pct_chg_nm')
        X = data_to_regress.loc[:, x_col]
        # 有些行业可能不在，删除相应的行业哑变量
        for fac in X.sum()[X.sum() == 0].index:
            if fac not in factors_name:
                del X[fac]

        X['constant'] = np.ones(len(X))
        # 回归并存储结果
        try:
            _, cur_factor_ret, resid_tmp = regress(y, X)
        except Exception as e:
            print('debug')

        resid_tmp.name = date
        resid = pd.concat([resid, resid_tmp], axis=1)
        cur_factor_ret.name = date
        factor_rets = pd.concat([factor_rets, cur_factor_ret], axis=1)

    factor_rets = factor_rets.T
    try:
        factor_rets.loc[added_index, :] = None
    except Exception as e:
        print('de')

    resid = resid.T
    return factor_rets, resid


# 根据历史的因子表现，得到预测的因子收益值
def forecast_factor_return(factors_name, factor_rets, window=12):
    # 对ROE_q以及growth因子的负值纠正为0
    factors_to_correct_over_zeros = ['growth', 'quality']
    factors_to_correct_tmp = [factor for factor in factors_name if factor in factors_to_correct_over_zeros]
    if len(factors_to_correct_tmp) > 0:
        for fac in factors_to_correct_tmp:
            try:
                fac_name = [f for f in factor_rets.columns if f.startswith(fac)][0]
            except IndexError:
                continue
            # 第一个为cond,第二个是cond为假的设定的值
            factor_rets[fac_name] = factor_rets[fac_name].where(factor_rets[fac_name] >= 0, 0.000001)

    # 指数加权权重-窗口平滑处理
    factor_rets = factor_rets.rolling(window=window, min_periods=1).mean().shift(1)
    factor_rets = factor_rets.dropna(how='all', axis=0)

    return factor_rets


def my_factor_return_forecast(factors_name, factors_panel,
                              window=12, half_life=6):
    """
    因子收益预测：
    输入：自变量(x)因子名称（,分隔），类型：字符串
    输出：截面回归得到的因子收益率预测值，行：因子名称，列：截面回归当期日期
    """

    # 截面回归，获取回归系数，作为因子收益，factor_rets的index里面包含alpha因子和行业因子
    factor_rets = pd.DataFrame()
    for date in sorted(factors_panel.keys()):
        data_to_regress = factors_panel[date]
        # 删除nan
        data_to_regress = data_to_regress.dropna(how='any', axis=0)
        if data_to_regress.empty:
            # 记录最后一个月的日期，在最后shift的时候添加上
            added_index = date
            continue

        # 分成X,y
        try:
            y = data_to_regress.loc[:, 'PCT_CHG_NM']
        except Exception as e:
            print('no pct chag nm')

        x_col = list(data_to_regress.columns)
        x_col.remove('PCT_CHG_NM')
        X = data_to_regress.loc[:, x_col]
        # 有些行业可能不在，删除相应的行业哑变量
        for fac in X.sum()[X.sum() == 0].index:
            if fac not in factors_name:
                del X[fac]

        X['constant'] = np.ones(len(X))
        # 回归并存储结果
        try:
            _, cur_factor_ret, _ = regress(y, X)
        except Exception as e:
            print('debug')

        cur_factor_ret.name = date
        factor_rets = pd.concat([factor_rets, cur_factor_ret], axis=1)

    factor_rets = factor_rets.T
    try:
        factor_rets.loc[added_index, :] = None
    except Exception as e:
        print('de')

    # 对ROE_q以及growth因子的负值纠正为0
    factors_to_correct_over_zeros = ['growth', 'quality', 'mom']
    factors_to_correct_tmp = [factor for factor in factors_name if factor in factors_to_correct_over_zeros]
    if len(factors_to_correct_tmp) > 0:
        for fac in factors_to_correct_tmp:
            try:
                fac_name = [f for f in factor_rets.columns if f.startswith(fac)][0]
            except IndexError:
                continue
            # 第一个为cond,第二个是cond为假的设定的值
            factor_rets[fac_name] = factor_rets[fac_name].where(factor_rets[fac_name] >= 0, 0.000001)

    # factors_to_correct_below_zeros = ['reverse', 'lip']
    # factors_to_correct_tmp = [factor for factor in factors_name if factor in factors_to_correct_below_zeros]
    # if len(factors_to_correct_tmp) > 0:
    #     for fac in factors_to_correct_tmp:
    #         try:
    #             fac_name = [f for f in factor_rets.columns if f.startswith(fac)][0]
    #         except IndexError:
    #             continue
    #         # 第一个为cond,第二个是cond为假的设定的值
    #         factor_rets[fac_name] = factor_rets[fac_name].where(factor_rets[fac_name] <= 0, -0.000001)

    # 指数加权权重-窗口平滑处理
    factor_rets = factor_rets.rolling(window=window, min_periods=1).mean().shift(1)
    factor_rets = factor_rets.dropna(how='all', axis=0)
    # factor_rets.empty
    return factor_rets


def forecast_factor_est_return(path_dict, params):         #曾用名：select_industry_model

    # save_path = path_dict['save_path']
    matrix_path = path_dict['matrix_path']

    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    factors, window, half_life = params['factors'], params['window'], \
                                 params['half_life']

    factors.extend(['PCT_CHG_NM'])
    # 得到所有因子的因子矩阵，以dict形式存储，行业部分没有风险因子
    factors_dict = {fac: get_factor(factors, basic_path=matrix_path)[fac] for fac in factors}
    # 将alpha因子整理为截面形式
    factors_panel = concat_factors_panel(factors=None, factors_dict=factors_dict, codes=None,
                                         ind=False, mktcap=False, perchg_nm=False)

    # 删除开始factor不全的截面
    to_del = []
    for key, values in factors_panel.items():
        print(key)
        for f in factors:
            if f not in values.columns:
                print(f)
                to_del.append(key)
                break

    for d in to_del:
        factors_panel.pop(d)

    # 因子收益预测模型
    est_fac_rets = my_factor_return_forecast(factors, factors_panel, window,
                                             half_life)
    factors.remove('PCT_CHG_NM')
    est_fac_rets = est_fac_rets.dropna(axis=0, how='any')
    try:
        est_fac_rets = est_fac_rets[factors]
    except Exception as e:
        print('asdfasdf')
    # 全部股票收益预测
    est_stock_rets = get_est_stock_return(factors, factors_panel,
                                          est_fac_rets, window, half_life)
    print('计算预期收益率完成...')

    est_stock_rets.name = 'est_industry_return'

    # est_stock_rets = est_stock_rets.fillna(0)

    return est_stock_rets


def concat_first_industry_rets(est_second_indus_rets):
    factors = ['industry_zx', 'MKT_CAP_FLOAT']
    factors_dict = {fac: get_factor(factors)[fac] for fac in factors}
    # 将行业和市值因子整理为截面形式
    factors_panel = concat_factors_panel(factors=None, factors_dict=factors_dict, codes=None,
                                         ind=False, mktcap=False, perchg_nm=False)

    # 对每一个panel,添加子行业的预期收益率,然后计算一级行业收益率
    est_first_indus_rets = pd.DataFrame(columns=est_second_indus_rets.columns)
    for date in sorted(factors_panel.keys()):
        if date not in est_second_indus_rets.columns:
            continue
        #date = sorted(factors_panel.keys())[1]
        factors_df = factors_panel[date]
        factors_df['est_rets'] = est_second_indus_rets.loc[:, date]
        factors_df = factors_df.fillna(0)
        grouped = factors_df.groupby(['industry_zx'])
        for group_name, group_df in grouped:
            w = group_df['MKT_CAP_FLOAT']/np.sum(group_df['MKT_CAP_FLOAT'])
            est_first_indus_rets.loc[group_name, date] = np.sum(w * group_df['est_rets'])

    est_first_indus_rets = est_first_indus_rets.drop(0)

    return est_first_indus_rets


def adjust_indus_wt(index_wt, est_indus_rets):
    # 调整权重的思路为，对预期收益最高的前20%的行业权重提高一倍，然后再对所有行业做归一化处理。
    adjust_wt = pd.DataFrame(index=est_indus_rets.index, columns=est_indus_rets.columns)

    for date in adjust_wt.columns:
        # 合并
        tmp_wei = pd.concat([index_wt[date], est_indus_rets[date]], axis=1)
        tmp_wei.columns = ['wt', 'rets']
        tmp_wei = tmp_wei.fillna(0)
        # 选择前20%的位置和后20%的位置
        sort_tmp = sorted(tmp_wei['rets'], reverse=True)
        l_top = int(0.2 * len(sort_tmp))
        l_bottom = int(0.8 * len(sort_tmp))
        # 按列遍历，大于前20%的权重翻倍，后20%缩小50%
        for i, row in tmp_wei.iterrows():
            if row['rets'] > sort_tmp[l_top]:
                row['wt'] = 2 * row['wt']
            if row['rets'] < sort_tmp[l_bottom]:
                row['wt'] = 0.5 * row['wt']
        # 归一化
        tmp_wei['wt'] = tmp_wei['wt'] / tmp_wei['wt'].sum()

        adjust_wt.loc[:, date] = tmp_wei['wt']

    return adjust_wt


def get_wei(est_rets, bestN):
    # 通过预测的行业收益，得到前N个行业的权重
    # columns为月份，index为各个行业
    # 逻辑：等权配置预测收益排名为前N的行业
    wt = pd.DataFrame(np.zeros(np.shape(est_rets)), index=est_rets.index, columns=est_rets.columns)
    # 逐列遍历
    for col, itmes in est_rets.iteritems():
        itmes = itmes.sort_values(ascending=False)
        select_id = itmes.index[0:bestN]
        wt.loc[select_id, col] = 1 / bestN

    return wt


def best_N_backtest(data_dict):
    # 简化逻辑：
    # 从预测收益率得到预期表现前N的行业。
    est_rets = data_dict['est_rets']
    bestN = data_dict['bestN']
    wt = pd.DataFrame(np.zeros(np.shape(est_rets)), index=est_rets.index, columns=est_rets.columns)
    # 逐列遍历
    for col, itmes in est_rets.iteritems():
        itmes = itmes.sort_values(ascending=False)
        select_id = itmes.index[0:bestN]
        wt.loc[select_id, col] = 1/bestN

    market_data = data_dict['market_data']
    benchmark = data_dict['benchmarkdata']
    start_date = est_rets.columns[0]
    end_date = est_rets.columns[-1]

    # # 根据优化得到的各月末截面期HS300成分股股票权重，进行回测
    bt = Backtest_stock(market_data=market_data,
                        start_date=start_date,
                        end_date=end_date,
                        benchmarkdata=benchmark,
                        stock_weights=wt,
                        refreshdays=[col for col in est_rets.columns],
                        use_pctchg=True)
    bt.run_backtest()
    print('回测结束, 进行回测结果分析...')
    summary_yearly = bt.summary_yearly()  # 回测统计


    # 业绩归因
    # p_attr = performance_attribution(factors_dict, index_wt, stock_wt,
    #                                  est_fac_rets, start_date, end_date)

    '''
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
    
    '''


# 统计预测表现与实际的行业表现得区别
def analysis(est_rets, real_rets):
    '''
    est_rets为预测得行业收益, real_rets 为实际得行业收益
    逻辑1：计算预测的准确率，预测是前25%，实际真的是25%，则表示准确。
    '''

    cumbasis = analysis_by_indus(est_rets, real_rets)
    # n = 25
    # res1 = analysis_by_rank(est_rets, real_rets, n)
    # cumbasis.to_csv(path, encoding='gbk')
    return cumbasis


def analysis_by_indus(est_rets, real_rets):
    # 逻辑2：计算预测行业收益与实际行业收益的差值，如预测5%，实际7%，差值为2%，实际3%，差值同样为2%

    basis = abs(real_rets - est_rets)
    basis = basis.dropna(axis=1)
    basis = 1 + basis/100
    cumbasis = basis.cumprod(axis=1)
    cumbasis = cumbasis.sort_values(by=[cumbasis.columns[-1]], ascending=False)

    return cumbasis


def analysis_by_rank(est_rets, real_rets, n):
    # 逻辑：计算预测行业位置的准确铝，如预测金融涨幅是前25%，真的是涨幅前25%，则预测准确。
    datelist = set(est_rets.columns) & set(real_rets.columns)
    datelist = sorted(list(datelist))

    win_rate = pd.DataFrame(columns=[str(n) + '分位数'])
    loc = int(np.percentile([i for i in range(0, len(est_rets))], abs(n)))
    for date in datelist:

        if n > 0:
            est_rets_tmp = est_rets[date].sort_values(ascending=False)
            real_rets_tmp = real_rets[date].sort_values(ascending=False)
        elif n < 0:
            est_rets_tmp = est_rets[date].sort_values()
            real_rets_tmp = real_rets[date].sort_values()

        est_indus = set(est_rets_tmp.index[0:loc])
        real_indus = set(real_rets_tmp.index[0:loc])

        right_set = est_indus & real_indus
        wr = len(right_set) / (loc + 1)
        win_rate.loc[date, str(n) + '分位数'] = wr

    plt.plot(win_rate)
    plt.legend(win_rate.columns)
    plt.show()


# 把字典里的str全部转换成小写字符
def lower_dict(f_dict):
    f_dict_lower = {}
    for k, values in f_dict.items():
        f_list = []
        for v in values:
            f_list.append(v.lower())
        f_dict_lower[k] = f_list

    return f_dict_lower


# 统计过去所有行业的年度收益率，每一个年度根据涨跌幅排序，分别列出行业和涨跌幅。
# 然后看看能不能设定下不同涨跌幅不同颜色。

def year_analysis():
    spath = r'D:\pythoncode\IndexEnhancement\行业多因子\second_industry\单因子检验\因子矩阵'

    market_data = get_matrix_data('PCT_CHG_NM', spath)['PCT_CHG_NM']
    # 存在大于1的数，表面未除以100，否则就是用百分数形式表达，不用变换
    if np.any(market_data > 1):
        market_data = 0.01 * market_data.fillna(0)
    else:
        market_data = market_data.fillna(0)
    market_data = market_data.shift(1, axis=1)
    market_data = market_data.dropna(how='all', axis=1)

    def tt(x):
        x = 1 + x
        x = x.cumprod() - 1
        return x[-1]

    year_return = pd.DataFrame()
    for name, row in market_data.iterrows():
        r = row.resample('y', how=tt)
        tmp = pd.DataFrame(r, columns=[name])
        year_return = pd.concat([year_return, tmp], axis=1)

    year_return = year_return.T
    year_return_resize = pd.DataFrame()
    for n, col in year_return.iteritems():
        col = col.sort_values(ascending=False)
        tmp_pd = pd.DataFrame({str(n.year) + '年：行业': col.index, str(n.year) + '年：涨幅': col.values})
        year_return_resize = pd.concat([year_return_resize, tmp_pd], axis=1)

    return year_return_resize



