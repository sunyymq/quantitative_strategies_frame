# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:20:21 2018

@author: admin
"""
import calendar
import os
import numpy as np
import pandas as pd
import pandas.tseries.offsets as toffsets
from sqlalchemy import text
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy import VARCHAR, Date, Column, Index, cast  # , UniqueConstraint,
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import insert
from pandas.core.groupby.groupby import DataError
import pymysql
import warnings

warnings.filterwarnings("ignore")
pymysql.install_as_MySQLdb()
# from WindPy import w
# w.start()

local_db = 'juyuan'

engines = {
    'src': ['JYDBFK', 'mssql+pymssql', 'yjs', 'yjs', '172.17.6.5', '42000'],  # src是公司的聚源数据库地址
          }


# 创建连接引擎，需手动修改用户名，密码，端口号（可直接用默认值），数据库名称
def create_db_connect(db_name=local_db, dbtype='mysql',user='zsl',
                      password='password', host='127.0.0.1', port='3306',
                      echo=True):
    engine = create_engine((f"{dbtype}://{user}:{password}"
                           f"@{host}:{port}/{db_name}?charset=utf8"),
                           encoding='gbk')
#    print(engine)
    return engine


def refresh_tables(engine):
    base = declarative_base()
    base.metadata.reflect(engine)
    stock_tables = base.metadata.tables
    return stock_tables


def code_fix(s):
    if len(s) < 6:
        s = '0' * (6-len(s)) +s
    return s


def get_stock_basicinfos(conn, update=True):
    global db_file_path
    code_table_name = 'secumain'
    selected_cols = 'InnerCode, CompanyCode, SecuCode, SecuAbbr, ListedDate, ListedState'
    judge_col = 'SecuCategory'
    judge_val = 1
    query_sql = f"select {selected_cols} from {code_table_name} where {judge_col} = {judge_val}"
    dat = pd.read_sql_query(query_sql, conn, index_col=['SecuCode'])

    codes = [code_fix(s) for s in dat.index
             if not s.startswith('X') and
             not s.startswith('8') and
             not s.startswith('4') and
             not s.startswith('S') and
             not s.startswith('9')]

    # 处理中文乱码问题
    for i in dat.index:
        if isinstance(dat.loc[i, 'SecuAbbr'], str) and not dat.loc[i, 'SecuAbbr'].isnumeric():
            dat.loc[i, 'SecuAbbr'] = dat.loc[i, 'SecuAbbr'].encode('latin-1').decode('gbk')

    # 上市状态的具体描述：1 - 上市，3 - 暂停，5 - 终止，9 - 其他。

    dat = dat.loc[codes]
    # 删除不上市的股票
    dat = dat.drop(dat.index[dat['ListedState'] != 1.0], axis=0)
    if update:
        dat.to_csv(os.path.join(db_file_path, 'stocks_codes.csv'), encoding='gbk')
    return dat


# 得到所有的股票代码，这个代码是在聚源里的代码
def get_code_range(col=None, if_index=False):
    global db_file_path
    if col is None:
        col = 'SecuCode'

    if if_index:
        basic_info = pd.read_excel(os.path.join(db_file_path, 'all_index_code.xlsx'),
                                   encoding='gbk')
        basic_info['SecuCode'] = [str(c) for c in basic_info['SecuCode']]
    else:
        basic_info = pd.read_csv(os.path.join(db_file_path, 'stocks_codes.csv'),
                                 engine='python', encoding='gbk')
        basic_info['SecuCode'] = [code_fix(str(c)) for c in basic_info['SecuCode']]

    if col == 'SecuCode':
        codes = basic_info[col].values
        return pd.Series(codes, index=codes)
    else:
        basic_info = basic_info.set_index([col])
        return basic_info['SecuCode']


def get_history_industry_citic(db_engine):
    global db_file_path
    tablename = 'LC_ExgIndustry'
    judge_col = 'CompanyCode'
    sdate_col, edate_col = 'InfoPublDate', 'CancelDate'

    codes = get_code_range(judge_col, if_index=False)
    trade_days = pd.read_excel(os.path.join(db_file_path, 'tradedays.xlsx'))
    trade_days = trade_days.values.flatten()  # flatten()用于将ndarry变成一个一维的数组
    all_days = pd.date_range(start='2003-01-01', end=str(pd.offsets.date.today()))

    local_tables = refresh_tables(db_engine)
    cur_table = local_tables[tablename]

    session = sessionmaker(bind=db_engine)
    sess = session()
    res = sess.query(cur_table).filter(cur_table.c['Standard'] == 3).with_entities(cur_table.c[judge_col],
                                                                                   cur_table.c[sdate_col],
                                                                                   cur_table.c[edate_col],
                                                                                   cur_table.c['FirstIndustryName'],
                                                                                   cur_table.c['SecondIndustryName'],
                                                                                   cur_table.c['ThirdIndustryName'])
    res = res.all()
    res = pd.DataFrame(res).set_index([judge_col])
    res = res.loc[codes.index, :]
    res = res.reset_index()
    codecs = [codes.loc[c] for c in res[judge_col].values]
    res[judge_col] = ['`' + str(c) for c in codecs]

    for index, row in res.iterrows():
        if isinstance(row['FirstIndustryName'], str) and not row['FirstIndustryName'].isnumeric():
            res.loc[index, 'FirstIndustryName'] = row['FirstIndustryName'].encode('latin-1').decode('gbk')
        if isinstance(row['SecondIndustryName'], str) and not row['SecondIndustryName'].isnumeric():
            res.loc[index, 'SecondIndustryName'] = row['SecondIndustryName'].encode('latin-1').decode('gbk')
        if isinstance(row['ThirdIndustryName'], str) and not row['ThirdIndustryName'].isnumeric():
            res.loc[index, 'ThirdIndustryName'] = row['ThirdIndustryName'].encode('latin-1').decode('gbk')

    for level in ['First', 'Second', 'Third']:
        level_ind = f'{level}IndustryName'
        save_dat = pd.DataFrame(index=trade_days)
        ind_chg_dat = res[[judge_col, sdate_col, level_ind]]
        for gp_name, gp_dat in ind_chg_dat.groupby(judge_col):
            gp_dat = gp_dat[[sdate_col, level_ind]].set_index(sdate_col)
            gp_dat = gp_dat.drop_duplicates()
            gp_dat = gp_dat.reindex(all_days)
            gp_dat = gp_dat.fillna(method='ffill')
            gp_dat = gp_dat.loc[trade_days]
            save_dat[gp_name] = gp_dat
        # save_dat.T.to_csv(os.path.join(db_file_path, level_ind+'.csv'),
        #                   encoding='gbk')
        print(f'Save {level_ind} success!')