import pandas as pd
import shelve
import numpy as np
import calendar
from collections import defaultdict
from utility.tool0 import Data
from datetime import datetime
from utility.tool2 import ma
from utility.constant import data_dair, root_dair
import os
from utility.constant import code_name_map_citic, code_name_map_sw, index_code_name_map
from utility.relate_to_tushare import stocks_basis, trade_days, generate_months_ends
from utility.template import SelectStockTemplate



def style_index():
    path = r'D:\定期报告\月报\巨潮风格指数.xlsx'
    dat = pd.read_excel(path, enconding='gbk')
    dat.set_index(dat.columns[0], inplace=True)

    mes = generate_months_ends()
    new_index = [i for i in dat.index if i in mes]
    dat = dat.loc[new_index, :]

    # 分别计算过去一个月、半年、一年的收益率
    res = pd.DataFrame()
    ret1 = dat.loc[dat.index[-1], :]/dat.loc[dat.index[-2], :] - 1
    ret6 = dat.loc[dat.index[-1], :] / dat.loc[dat.index[-7], :] - 1
    ret12 = dat.loc[dat.index[-1], :] / dat.loc[dat.index[-13], :] - 1

    res = pd.concat([res, pd.DataFrame({'过去1个月': ret1})], axis=1)
    res = pd.concat([res, pd.DataFrame({'过去6个月': ret6})], axis=1)
    res = pd.concat([res, pd.DataFrame({'过去12个月': ret12})], axis=1)

    save_path = r'D:\定期报告\月报\巨潮风格指数收益情况.xlsx'
    res.to_excel(save_path)

    return res
    dat


if __name__ == '__main__':
    style_index()


