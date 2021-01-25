import copy
import os
from utility.tool0 import Data
from utility.constant import data_dair
from juyuan.connect_to_juyuan import *

save_path = os.path.join(data_dair, 'download_from_juyuan')


def add_postfix(d_df):
    new_index = []
    for i in d_df.index:
        si = str(i)
        if len(si) < 6:
            si = '0' * (6-len(si)) + si
        # 6开头的， .SH
        # 000; 300开头的 .SZ
        if si.startswith('6'):
            si = si + '.SH'
            new_index.append(si)
        elif si.startswith('0') or si.startswith('3'):
            si = si + '.SZ'
            new_index.append(si)

    d_df.index = new_index
    return d_df


# 把列表里的字符串连成一个字符串并于","连接
def str_from_list(f_list):
    vl = ''
    for f in f_list:
        if len(vl) == 0:
            vl = f
        else:
            vl = vl + ', ' + f
    return vl


def change_shape(dat_df, basic_info):
    if 'CompanyCode' in dat_df.columns:
        changed = 'CompanyCode'
    elif 'InnerCode' in dat_df.columns:
        changed = 'InnerCode'

    tmp_info = basic_info[[changed, 'SecuCode']]
    tmp_info[changed] = tmp_info[changed].apply(lambda x: np.int(x))

    res = copy.deepcopy(dat_df)
    res[changed] = res[changed].apply(lambda x: np.int(x))

    # 换日期的名字
    if 'EndDate' in res.columns:
        new_cols = 'EndDate'
    elif 'TradingDay' in res.columns:
        new_cols = 'TradingDay'
    elif 'SuspendDate' in res.columns:
        new_cols = 'SuspendDate'

    left_v = set(res.columns) - set([new_cols]) - set([changed])
    if len(left_v) == 0:
        res['left'] = 1
        left_v = ['left']

    dat = res.pivot_table(index=changed, columns=new_cols,
                          values=list(left_v)[0])

    tmp_info = tmp_info.set_index(changed)
    dat['SecuCode'] = None
    for i in dat.index:
        if i in tmp_info.index:
            dat.loc[i, 'SecuCode'] = tmp_info.loc[i, 'SecuCode']

    dat['SecuCode'] = dat['SecuCode'].fillna(0)
    dat.drop(dat.index[dat['SecuCode'] == 0], axis=0, inplace=True)
    dat.set_index('SecuCode', inplace=True)

    # try:
    #     res[changed].replace(tmp_info[changed].values, tmp_info['SecuCode'].values) #, inplace=True)
    # except:
    #     for i in tmp_info.index:
    #         res[changed].replace(tmp_info.loc[i, changed], tmp_info.loc[i, 'SecuCode'], inplace=True)
    # res.rename(columns={changed: 'SecuCode'}, inplace=True)
    # res.head()

    return dat


def down_from_juyuan(db_engine, basic_info, factor, basic_str, tablename, statdate_str='2008-01-01'):
    sql = basic_str + ', ' + factor
    query_sql = f"select {sql} from {tablename}"

    if tablename in ['LC_BalanceSheetAll', 'LC_IncomeStatementAll']:
        query_sql = query_sql + ' where IfMerged=1 and IfAdjusted=2'

    if tablename == 'QT_Performance' or tablename == 'QT_DailyQuote':
        query_sql = query_sql + r" where TradingDay > CONVERT(DATETIME, '{}', 102) ".format(statdate_str)

    if factor == 'TurnoverDeals':
        query_sql = query_sql + r'and TurnoverDeals <> 0'

    try:
        dat = pd.read_sql_query(query_sql, db_engine)
    except Exception as e:
        print('deg')

    # 删除掉非股票类标的
    if 'CompanyCode' in dat.columns:
        selected = dat['CompanyCode'].apply(
            lambda x: True if np.float(x) in basic_info['CompanyCode'].values else False)
        dat.drop(selected.index[selected == False], inplace=True)

    dat = change_shape(dat, basic_info)

    # 选择06年以后的数据
    new_cols = [col for col in dat.columns if col.to_pydatetime().year >= 2006]
    dat = dat[new_cols]

    dat = add_postfix(dat)

    return dat


# 给一个excel的路径，得到最后一列的日期。用来知道上次更新到什么时候了
def last_date(path):
    pass


# 把一个从聚源下载的更新数据与原来有的csv数据合并
def concat_df_with_csv(dat_df, path):
    pass


def update_all_basic():

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    factor_dict = {
                   # 'LC_BalanceSheetAll':              # 资产负债表
                   #      ['CompanyCode', 	          # 公司代码	int	否	注2
                   #       'EndDate',                   # 截止日期
                   #       'TotalAssets',               # 总资产
                   #       'TotalShareholderEquity',    # 所有者权益合计
                   #       'TotalLiability',            # 负债合计
                   #       'GoodWill',                  # 商誉
                   #      ],
                   #
                   # 'LC_MainIndexNew':
                   #      ['CompanyCode',
                   #       'EndDate',
                   #       'BasicEPS',                 # 基本每股收益
                   #       'EPSTTM',
                   #       'NetAssetPS',               # 每股净资产
                   #       'TotalOperatingRevenuePS',  # 每股营业总收入
                   #       'ROE',
                   #       'ROETTM',
                   #       'ROA',
                   #       'ROATTM',
                   #       'NetProfitRatio',
                   #       'NetProfitRatioTTM',         # 销售净利润TTM
                   #       'OperatingRevenuePSTTM',     # 每股营业收入
                   #       'NetProfitRatio',             # 销售净利率(%)
                   #       'NetProfitRatioTTM',          # 销售净利率_TTM( %)
                   #       'GrossIncomeRatio',           # 销售毛利率( %)
                   #       'GrossIncomeRatioTTM',       # 销售毛利率
                   #       'DebtAssetsRatio',           # 资产负债率
                   #       'CashRateOfSalesTTM',        # 经营活动产生的现金流量净额/营业收入_TTM(%)
                   #       'AdminiExpenseRate',         # 管理费用/营业总收入(%)
                   #       'NetProfit',                 # 归属母公司净利润(元)
                   #       'NetProfitCut',              # 扣除非经常性损益后的净利润(元)
                   #
                   #       'BasicEPSYOY',           # 基本每股收益同比增长率
                   #       'OperProfitGrowRate',    # 营业利润同比增长率
                   #       'NetProfitGrowRate',     # 净利润同比增长率
                   #       'NetProfitGrowRate',    # 净利润同比增长率
                   #       'NAORYOY',              # 净资产收益率(摊薄)同比增长(%)
                   #       'OperCashPSGrowRate',    # 每股经营活动产生的现金流量净额同比增长(%)
                   #
                   #       # 运营能力
                   #       'InventoryTRate',      # 存货周转率(次)
                   #       'ARTRate',             # 应收账款周转率(次)
                   #       'CurrentAssetsTRate',  # 流动资产周转率(次)
                   #       'FixedAssetTRate',     # 固定资产周转率(次)    decimal(18, 4)        注87
                   #       'TotalAssetTRate',     # 总资产周转率(次)
                   #      ],
                   # 'QT_DailyQuote':
                   #      [
                   #        'InnerCode',
                   #        'TradingDay',
                   #        'OpenPrice',         # 今开盘(元)    decimal(10, 4)
                   #        'HighPrice',         # 最高价(元)    decimal(10, 4)
                   #        'LowPrice',          # 最低价(元)    decimal(10, 4)
                   #        'ClosePrice',        # 收盘价(元)    decimal(10, 4)
                   #        'TurnoverVolume',    # 成交量(股)    decimal(20, 0)        注2
                   #        'TurnoverValue',     # 成交金额(元)    decimal(19, 4)
                   #        'TurnoverDeals',     # 成交笔数(笔)
                   #    ],
                   'QT_Performance':
                       [
                           'InnerCode',
                           'TradingDay',
                           'OpenPrice',  # 今开盘(元)    decimal(10, 4)
                           'HighPrice',  # 最高价(元)    decimal(10, 4)
                           'LowPrice',  # 最低价(元)    decimal(10, 4)
                           'ClosePrice',  # 收盘价(元)    decimal(10, 4)
                           'TurnoverVolume',  # 成交量(股)    decimal(20, 0)        注2
                           'TurnoverValue',  # 成交金额(元)    decimal(19, 4)
                           'ChangePCT',
                       ],

                   # 'LC_IncomeStatementAll':      # 利润分配表
                   #     [
                   #         'CompanyCode',
                   #         'EndDate',
                   #         'TotalOperatingRevenue',  # 营业总收入
                   #         'OperatingRevenue',       # 营业收入
                   #         'OperatingCost',          # 营业成本
                   #         'NetProfit',              # 净利润
                   #         'RAndD',                  # 研发费用
                   #     ],
                   }

    basic_list = ['CompanyCode', 'InnerCode', 'EndDate', 'TradingDay', 'ReportDate', 'InvestType']    # 'IfMerged', 'IfAdjusted'

    db_engine = create_db_connect(*engines['src'])
    print(db_engine)

    # 刷新所有的表
    src_tables = refresh_tables(db_engine)
    basic_info = get_stock_basicinfos(db_engine, update=False)
    # basic_info[]
    basic_info['SecuCode'] = basic_info.index
    basic_info = basic_info.dropna(axis=0, how='any')
    basic_info.index = range(0, len(basic_info))

    basic_info.sort_values('CompanyCode')

    # 存储进CSV，使得Data类可以访问
    bi = copy.deepcopy(basic_info)
    bi = bi.set_index('SecuCode')
    bi = add_postfix(bi)
    bi.to_csv(os.path.join(save_path, 'basic_info.csv'), encoding='gbk')

    for tablename, value in factor_dict.items():
        # 把value里的数据分为基本数据和变量数量两大类
        basic_tmp = []
        factor_names = []
        for f in value:
             if f in basic_list:
                 basic_tmp.append(f)
             else:
                 factor_names.append(f)

        basic_str = str_from_list(basic_tmp)

        for factor in factor_names:
            print(factor)
            dat = down_from_juyuan(db_engine, basic_info, factor, basic_str, tablename, statdate_str='2005-01-01')
            if tablename in ['QT_Performance', 'QT_DailyQuote']:
                dat.to_csv(os.path.join(save_path, factor + '_daily.csv'), encoding='gbk')
            else:
                dat.to_csv(os.path.join(save_path, factor+'.csv'), encoding='gbk')


def compute_changepect_open_daily():
    data = Data()
    open_daily = data.openprice_daily
    adjfactor = data.adjfactor
    open_daily = open_daily*adjfactor
    open_daily.dropna(axis=1, how='all', inplace=True)

    open_daily_shift = open_daily.shift(1, axis=1)
    changepect_open_daily = open_daily/open_daily_shift
    changepect_open_daily = (changepect_open_daily - 1)*100
    changepect_open_daily.dropna(how='all', axis=1, inplace=True)
    data.save(changepect_open_daily, 'changepect_open_daily'.upper())


if __name__ == '__main__':
    update_all_basic()
    # compute_daily_value()
    # compute_month_value()




