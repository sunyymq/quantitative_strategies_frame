#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import statsmodels.api as sm
from datetime import datetime
import shutil
from functools import reduce
from sklearn.covariance import LedoitWolf
from utility.factor_data_preprocess import add_to_panels, align
from utility.tool0 import Data, add_stock_pool_txt
from utility.tool3 import adjust_months, append_df, wr_excel
from utility.relate_to_tushare import stocks_basis, generate_months_ends, trade_days
from utility.single_factor_test import get_firt_industry_list

from utility.stock_pool import financial_condition_pool, factor_condition_pool, concat_stock_pool, save_each_sec_name,\
    del_industry, keep_industry, get_scores, twice_sort, del_market, keep_market, from_stock_wei_2_industry_wei,\
    compute_icir, pattern_filter, index_filter
from utility.download_from_wind import section_stock_infor
from utility.analysis import BackTest, bool_2_ones
from utility.stock_pool import get_scores_with_wei, get_scores, month_return_compare_to_market_index
from utility.select_industry import my_factor_concat, history_factor_return,  forecast_factor_return, \
    copy_matrix, forecast_factor_return
from utility.index_enhance import linear_programming, concat_factors_panel, get_factor, get_est_stock_return
from utility.optimization import optimization_fun
from utility.constant import data_dair, root_dair, default_dict, factor_dict_to_concate  #,factor_dict_for_scores,
from utility.single_factor_test import get_datdf_in_panel, panel_to_matrix
from app.from_stock_pool_class import growth_stock_pool

import matplotlib
matplotlib.use('Agg')


if '__main__' == __name__:

    # 分行业遍历
    indus_list = get_firt_industry_list()
    # for indus in indus_list:
    if True:
        indus = '食品饮料'
        print(indus)

        para_set_mud = 'ei'              # 'regular', 'my_way'
        # 参数组合说明与设置：
        # 2，对于我的主要研究方向，不同行业使用不同的基本面因子选股的策略，检验配置的参数为：
        method = 'score'
        score_mode = 'each_industry'
        special_market_dict = None
        use_risk_model = False
        select_type = None
        bt_or_latest = 'bt'
        bm = indus
        indus_d = {'handle_type': 'keep',
                   'to_handle_indus': [indus]
                   }
        my_freq = 'M'

        financial_dict = {'all': {'scope_0': ('roettm', -90, np.nan),
                                  }
                          }
        para_dict = None
        risk_factor = None

        if indus == indus_list[0]:
            update_ornot = 'renew'      # 'update'
        else:
            update_ornot = 'update'     # 'renew'  'update'

        start_date = datetime(2010, 1, 1)

        # 只保留default和指定的行业
        fd = {}
        for key, values in factor_dict_to_concate.items():
            if key in [indus, 'default']:
                fd.update({key: list(values.keys())})

        percent_n = 0.1
        if 'bm' not in dir():
            bm = None
        if 'rps_para' not in dir():
            rps_para = None
        if 'use_risk_model' not in dir():
            use_risk_model = False
        if 'risk_factor' not in dir():
            risk_factor = None

        res = growth_stock_pool(method=method, score_m=score_mode, select_type=select_type,
                                risk_model=use_risk_model, bt_or_latest=bt_or_latest, freq=my_freq,
                                risk_factor=risk_factor, special_market_dict=special_market_dict,
                                update_ornot=update_ornot, para_d=para_dict, indus_d=indus_d,
                                bm=bm, start_d=start_date, fd_for_scores=fd, percent_n=percent_n,
                                rps_para=rps_para, financial_dict=financial_dict)

        save_path = r'D:\Database_Stock\临时'
        if bt_or_latest == 'bt':                # 'bt' , 'latest'
            res, nv, each_year, fig, pool_sec, latest_wei = res[0], res[1], res[2], res[3], res[4], res[5]

            p = os.path.join(save_path, '分行业测试', indus)
            if not os.path.exists(p):
                os.makedirs(p)

            res.to_csv(os.path.join(p, '指标.csv'), encoding='gbk')
            nv.to_csv(os.path.join(p, '净值.csv'), encoding='gbk')
            fig.savefig(os.path.join(os.path.join(p, indus+'净值走势图.png')))
            plt.close()
            each_year.to_csv(os.path.join(p, '历年表现.csv'), encoding='gbk')
            pool_sec.to_csv(os.path.join(p, '历期股票池.csv'), encoding='gbk')
            latest_wei.to_csv(os.path.join(p, '最近一期股票池.csv'), encoding='gbk')

        elif bt_or_latest == 'latest':
            newest_pool, stock_pool = res[0], res[1]
        elif bt_or_latest == 'only_pool':
            stock_pool = res
            stock_pool.to_csv(os.path.join(save_path, '历史每期股票池_week.csv'), encoding='gbk')









