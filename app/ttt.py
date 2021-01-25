from datetime import datetime
import pandas as pd
import numpy as np
import os
from utility.tool0 import Data

# data = Data()
# basic = data.stock_basic_inform
# p = r'D:\Database_Stock\因子预处理模块\因子'
# dir = os.listdir(p)
# fn = dir[-1]
#
# dat = pd.read_csv(os.path.join(p, fn), encoding='gbk')
# dat.set_index('Code', inplace=True)
#
# dat = pd.concat([dat, basic['申万三级行业']], axis=1, join='inner')
#
# first_list = ['农林牧渔', '电子', '化工', '计算机', '传媒', '轻工制造', '通信', '医药生物']
# for first_n in first_list:
#     print(first_n)
#     dat_tmp = dat.loc[dat['Industry_sw'] == first_n, :]
#     grouped = dat_tmp.groupby('申万三级行业')
#
#     se = pd.Series()
#     for i, v in grouped:
#         tt = pd.Series(data=v['Rdtosales'].mean(), index=[i])
#         se = pd.concat([se, tt])
#
#     res = pd.DataFrame({first_n: se})
#     res.to_csv(r'D:\Database_Stock\临时\Rdtosales_' + first_n + '_子行业统计.csv', encoding='gbk')

p = r'D:\Database_Stock\多因子选股\新合成因子\icir.csv'
icir = pd.read_csv(p, encoding='gbk')
icir.set_index(icir.columns[0], inplace=True)

se = icir.iloc[0, :]
abs(se)/np.sum(abs(se))

icir.apply(lambda x: abs(x)/np.sum(abs(x)), axis=1)


