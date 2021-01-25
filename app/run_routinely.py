import pandas as pd
import numpy as np
import os
from datetime import datetime
from utility.tool0 import Data
from app.data_management_class import Data_Management
from app.industry_select import generate_indus_wei, IndustrySelect
from factor_compute_and_update.factor_compute import compute_factor
from factor_compute_and_update.factor_update import generate_new_factor_table
from app.factor_process import process_fun
from app.from_stock_pool_class import growth_stock_pool
from utility.constant import data_dair, root_dair, default_dict, factor_dict_to_concate


# 常规运行一遍，更新基础数据、因子、策略表现情况，可以是周、可以是月
# 如果是周，仅更新基础数据，股票池表现情况，如果是月，则更新因子值
def run_routinely(period='M'):

    # 更新日度数据
    data_manage = Data_Management()
    data_manage.update_market_quote_daily()
    # print('日度基础数据更新完毕！')

    if period == 'M':
        # # 更新月度数据
        # data_manage.update_from_wind_monthly()
        # # 可能更新季度数据
        # data_manage.update_financial_data()

        # # 更新因子值
        # compute_factor('update')
        # 更新因子表
        # generate_new_factor_table()

        # # 因子预处理, 第一步
        # is_update = True
        # test_type = 'stock'
        # process_fun(test_type, is_update)
        # # 因子预处理, 第二步
        # test_type = 'each_industry'
        # process_fun(test_type, is_update)

        # 行业轮动策略更新
        # ise = IndustrySelect()
        # ise.set_industry_level(level=1)
        # ise.compose_way()
        # ise.select_indus()
        # ise.show_newest()

        # 更新选股策略表现情况
        # 选股策略参数
        indus_d = {'handle_type': 'delete',
                   'to_handle_indus': ['国防军工', '银行', '非银金融', '钢铁']
                   }
        method = 'score'
        special_market_dict = None
        score_mode = 'rps'
        use_risk_model = False
        select_type = 'total_num'
        start_date = datetime(2012, 1, 1)
        my_freq = 'M'
        bt_or_latest = 'latest'  # 'bt' , 'latest' ，'only_pool'
        update_ornot = 'update'  # 'renew'  'update'
        rps_para = 85

        financial_dict = {'all': {'scope_0': ('roettm', 0, np.nan),
                                  'scope_1': ('basicepsyoy', 0, 500),
                                  'rise_0': ('netprofitgrowrate', 2)
                                  }
                          }

        para_dict = None
        risk_factor = None

        concate_dict = factor_dict_to_concate
        fd = {}
        for key, values in factor_dict_to_concate.items():
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
                                rps_para=rps_para, financial_dict=financial_dict, concate_dict=concate_dict)

        newest_pool = res
        newest_pool.to_csv(r'D:\Database_Stock\股票池_最终\基本面加RPS最新股票池.csv', encoding='gbk')

    elif period == 'W':
        # 更新策略表现情况
        0


if __name__ == '__main__':
    run_routinely(period='M')

