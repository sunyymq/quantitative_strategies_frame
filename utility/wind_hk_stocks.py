import pandas as pd
import numpy as np
from collections import defaultdict
from utility.tool0 import Data
from utility.constant import data_dair
import os
from utility.constant import code_name_map_citic, code_name_map_sw, index_code_name_map
from utility.relate_to_tushare import stocks_basis, trade_days, generate_months_ends
from WindPy import *
from iFinDPy import *
from utility.constant import data_dair

save_path = os.path.join(data_dair, 'hk_stocks')


def hk_basic_inform():
    w.start()
    # 全部港股的股票代码
    res = w.wset("sectorconstituent", "date=2020-09-02;sectorid=a002010100000000", usedf=True)
    res_df = res[1]

    basic_df = pd.DataFrame()
    # SHSC 是否沪港通买入标的, SHSC2 是否深港通买入标的，industry_HS所属恒生行业名称
    for code in res_df['wind_code']:
        tmp_df = w.wsd(code, "sec_name,,ipo_date,sec_status,trade_code,windcode,SHSC,SHSC2,industry_HS",
                       "2020-09-01", "2020-09-01", "category=1", usedf=True)
        tmp_df = tmp_df[1]

        basic_df = pd.concat([basic_df, tmp_df], axis=0)
    basic_df.index.name = 'Code'
    basic_df.to_csv(os.path.join(save_path, 'basic_inf.csv'), encoding='gbk')
    return basic_df


def report_dates():
    w.start()
    res_df = w.tdays("2010-01-01", "2020-09-02", "Days=Alldays;Period=S", usedf=True)
    res_df = res_df[1]

    return list(res_df.index)


if __name__ == '__main__':
    # hk_basic_inform()
    report_dates()

    thsDataBasicData = THS_BasicData('600000.SH,600004.SH', 'ths_stock_short_name_stock;ths_ipo_date_stock', ';')
    basicData = THS_Trans2DataFrame(thsDataBasicData)

