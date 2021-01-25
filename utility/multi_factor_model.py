# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:07:47 2019

@author: HP
"""
import os
from datetime import datetime
import pandas as pd
import numpy as np
from utility.select_stock import optimization, optimization_equalwei, select_stock_ToF, code2other,\
                                 record_stock_and_industry, new_record_stock
from utility.select_industry import my_factor_concat, Symmetry, forecast_factor_est_return, copy_matrix, lower_dict
from utility.index_enhance import get_factor_corr, plot_corr_heatmap, performance_attribution, \
                                  get_factor, get_matrix_data
import shutil

corr_test_and_plot = False            # 是否进行相关性测试，在测试时需要，在跟随的时候不需要
sn = '因子热力图'                     # 画图的时候存储的名称


def multi_factor_model(factors_to_concat=None, path_dict=None, just_forecast=False):
    '''
    :param just_forecast: 适用于调试代码时，前期已经合成了大类因子的情况，不再做费时间的因子合成和正交处理，直接预测股票收益
    :return:
    '''

    if not factors_to_concat:
        factors_dict = {
                            # 'vol': ['std_12m', 'std_6m', 'std_3m', 'std_1m'],
                            'mom': ['Return_12m', 'Return_1m', 'Return_3m', 'Return_6m'],
                            # 'liq': ['STOQ_Barra', 'STOM_Barra', 'STOA_Barra'],
                            'quality': ['Roa_q', 'Roe_q'],
                            'value': ['Ep'],
                            'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q'],
                            'size': ['Lncap_barra'],
                            }

    if not path_dict:
        # 因子合成地址
        path_dict = {'save_path': r'D:\pythoncode\IndexEnhancement\多因子选股',
                     'factor_panel_path': r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）',
                     'ic_path': r'D:\pythoncode\IndexEnhancement\单因子检验\ic.csv',
                     'old_matrix_path': r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵'}

    # ----------------------------------
    # 因子检测
    f_list = []
    for values in factors_dict.values():
        for v in values:
            f_list.append(v)
    f_list = [col.replace('_div_', '/') for col in f_list]

    tmp = os.listdir(path_dict['factor_panel_path'])
    df_tmp = pd.read_csv(os.path.join(path_dict['factor_panel_path'], tmp[-1]),
                         engine='python', encoding='gbk', index_col=[0])

    cols = [col for col in df_tmp.columns]

    if not set(f_list).issubset(cols):
        print('factor 不够, 缺失的因子为：')
        print(set(f_list) - set(cols))
    else:
        print('通过因子完备性测试')

    if not just_forecast:
        # ----------------------------------
        # 因子合成
        if os.path.exists(os.path.join(path_dict['save_path'], '新合成因子')):
            shutil.rmtree(os.path.join(path_dict['save_path'], '新合成因子'))
        print('开始进行因子合成处理.....')
        for factor_con, factors_to_con in factors_dict.items():
            # 'equal_weight'   'max_ic_ir'
            my_factor_concat(path_dict, factors_to_con, factor_con, concat_type='equal_weight')
        print('因子合成完毕！')

        if corr_test_and_plot and sn:
            # 处理前的因子相关性
            factors_to_corr1 = []
            for value_list in factors_dict.values():
                factors_to_corr1.extend(value_list)

            ori_factor_path = r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵'
            corr_ori = get_factor_corr(factors_to_corr1, basic_path=ori_factor_path)
            plot_corr_heatmap(corr_ori, preprocessed=False, save_name=sn)
            # 处理后的因子相关性
            factors_to_corr2 = [value for value in factors_dict.keys()]
            concat_factor_path = r'D:\pythoncode\IndexEnhancement\多因子选股\新合成因子\因子矩阵'
            corr = get_factor_corr(factors_to_corr2, basic_path=concat_factor_path)
            plot_corr_heatmap(corr, preprocessed=True)
            print("相关系数热力图绘制完毕...")

        # # ----------------------------------
        # # 因子正交
        # print('开始进行因子正交处理...')
        # to_sym_factors = [k for k in factors_dict.keys()]
        #
        # path_dict = {'save_path': r'D:\pythoncode\IndexEnhancement\多因子选股',
        #              'factor_panel_path': r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子（已预处理）',
        #              'ic_path': r'D:\pythoncode\IndexEnhancement\单因子检验\ic.csv',
        #              'old_matrix_path': r'D:\pythoncode\IndexEnhancement\单因子检验\因子矩阵'}
        #
        # Symmetry(to_sym_factors, path_dict)
        # print('因子正交处理完毕')

    params = {
               'factors': [key for key in factors_dict.keys()],
               'window': 6,
               'half_life': None,
               }

    copy_matrix(path_dict['old_matrix_path'],
                os.path.join(path_dict['save_path'], '新合成因子', '因子矩阵'))

    # 估计预期收益
    path_dict.update({'matrix_path': os.path.join(path_dict['save_path'], '新合成因子', '因子矩阵')})
    est_stock_rets = forecast_factor_est_return(path_dict, params)

    return est_stock_rets


