'''
从wind数据库中提取数据，该文件主要提取上证综指的月度成分及权重。
'''

import pandas as pd
import shelve
import numpy as np
import calendar
from collections import defaultdict
from collections import defaultdict
from utility.tool0 import Data
from utility.tool2 import ma
from utility.constant import data_dair
import os
from utility.constant import code_name_map_citic, code_name_map_sw, index_code_name_map
from utility.relate_to_tushare import stocks_basis, trade_days, generate_months_ends, generate_calendar_month_ends
from WindPy import *


# w.wsd("000002.SZ", "est_instnum", "2020-07-28", "2020-08-26", "year=2020;Period=M")

class StructWind:
    def __init__(self, name, period, other=None, para_str=None, para=None):
        self.name = name
        self.period = period
        self.other = other
        self.para = para
        self.para_str = para_str

    def get_str(self):
        if self.other:
            oth = 'Period=' + self.period + ';' + self.other
        else:
            oth = 'Period=' + self.period

        if self.para:
            oth = oth + ';' + self.para_str + str(self.para)

        return oth


def update_stock_basic_inform():
    w.start()
    date_str = datetime.today().strftime('%Y-%m-%d')

    all_stock_codes = w.wset("sectorconstituent", "date=" + date_str + ";sectorid=a001010100000000", usedf=True)
    all_stock_codes = all_stock_codes[1]

    codes_str = ''
    for c in all_stock_codes['wind_code']:
        codes_str = codes_str + c + ','

    tds = trade_days()
    d_str = tds[-1].strftime('%Y%m%d')

    dat_df = w.wss(codes_str, "sec_name,ipo_date,mkt,sec_status,delist_date, industry_sw,industry_citic",
                   "tradeDate=" + d_str + ";industryType=4", usedf=True)

    dat_df = dat_df[1]

    res_df1 = pd.DataFrame()
    for key, se in dat_df.iterrows():
        citics = se['industry_citic'.upper()].split('--')
        if len(citics) == 3:
            tmp_df = pd.DataFrame(columns=[key], data=citics, index=['中信一级行业', '中信二级行业', '中信三级行业']).T
        else:
            print(se['industry_citic'.upper()])
            tmp_df = pd.DataFrame(columns=[key], index=['中信一级行业', '中信二级行业', '中信三级行业']).T

        res_df1 = pd.concat([res_df1, tmp_df], axis=0)

    res_df2 = pd.DataFrame()
    for key, se in dat_df.iterrows():
        try:
            citics = se['industry_sw'.upper()].split('-')
            if len(citics) == 3:
                tmp_df = pd.DataFrame(columns=[key], data=citics, index=['申万一级行业', '申万二级行业', '申万三级行业']).T
            else:
                print(se['industry_sw'.upper()])
                tmp_df = pd.DataFrame(columns=[key], index=['申万一级行业', '申万二级行业', '申万三级行业']).T

            res_df2 = pd.concat([res_df2, tmp_df], axis=0)
        except Exception as e:
            pass

    res = pd.concat([dat_df[['SEC_NAME', 'ipo_date'.upper(), 'mkt'.upper(), 'sec_status'.upper(), 'delist_date'.upper()]],
                     res_df1, res_df2], axis=1)
    res.index.name = 'CODE'
    save_path = os.path.join(data_dair, 'basic')
    res.to_csv(os.path.join(save_path, 'stock_basic_inform.csv'), encoding='gbk')

    # test = res['中信三级行业'].drop_duplicates()
    print('中信行业数据提取完毕')

    res_style = pd.DataFrame()
    # 得到风格指数的代码
    style_code = w.wset("sectorconstituent", "date=" + date_str + ";sectorid=1000006252000000", usedf=True)
    style_code = style_code[1]
    # style_code['wind_code']
    for i, se in style_code.iterrows():
        dat_tmp = w.wset("sectorconstituent", "date=" + date_str + ";windcode=" + se['wind_code'], usedf=True)
        dat_tmp = dat_tmp[1]
        dat_tmp['style_citic'] = se['sec_name'].split('(')[0]
        res_style = pd.concat([res_style, dat_tmp], axis=0)

    res_style.drop('date', axis=1, inplace=True)
    res_style.set_index('wind_code', inplace=True)
    res_style.index.name = 'CODE'
    res_style.to_csv(os.path.join(save_path, 'style_citic.csv'), encoding='gbk')

    w.close()
    print('个股信息跟新完毕')
    # todo:保存


# 提取指数成分和权重，并存储为月度数据
def get_composi_and_wei(panel_path):
    w.start()
    date = w.tdays("2009-01-02", "2019-08-02", "Period=M", usedf=True)
    date = date[1]
    date_list = [pd.Timestamp(d[0]).to_pydatetime() for d in date.values]

    for date in date_list:
        com_wei = w.wset("indexconstituent","date={};windcode=000001.SH".format(date.strftime("%Y-%m-%d")), usedf=True)
        com_wei = com_wei[1]
        if len(com_wei.index):
            com_wei.to_csv(os.path.join(panel_path, date.strftime("%Y-%m-%d")+'.csv'),  encoding='gbk')

    print('wind数据提取完毕')
    w.close()


def form_stock2_second_indus(panel_path, save_path):
    '''
    :param panel_path: 月度数据的存储地址
    :param save_path:  目标文件的存储地址
    :return:
    '''
    # 把股票的月度数据转换为行业的形式
    data = Data()
    indus_infor = data.secondindustryname
    indus_infor = data.reindex(indus_infor)

    dirlist = os.listdir(panel_path)

    indux_wei_total = pd.DataFrame()
    for f in dirlist:
        stock_wei = pd.read_csv(os.path.join(panel_path, f), encoding='gbk', engine='python')

        stock_wei = stock_wei.set_index('wind_code')
        if f.split('.')[0] in indus_infor.columns:
            stock_wei['second_indus'] = indus_infor[f.split('.')[0]]
        else:
            stock_wei['second_indus'] = indus_infor[indus_infor.columns[-1]]

        stock_wei = stock_wei.dropna(axis=0, how='any')
        stock_wei['i_weight'] = 100*stock_wei['i_weight']/np.sum(stock_wei['i_weight'])

        grouped = stock_wei[['i_weight', 'second_indus']].groupby('second_indus')
        indus_wei = grouped.sum()
        indus_wei = indus_wei.T
        indus_wei.index = [f.split('.')[0]]

        indux_wei_total = pd.concat([indux_wei_total, indus_wei], axis=0)
        indux_wei_total = indux_wei_total.fillna(0)

    indux_wei_total.to_csv(os.path.join(save_path, '二级行业权重.csv'),  encoding='gbk')


def form_stock2_first_indus(panel_path, save_path):
    '''
    :param panel_path: 月度数据的存储地址
    :param save_path:  目标文件的存储地址
    :return:
    '''
    # 把股票的月度数据转换为行业的形式
    data = Data()
    indus_infor = data.firstindustryname
    indus_infor = data.reindex(indus_infor)

    dirlist = os.listdir(panel_path)

    indux_wei_total = pd.DataFrame()
    for f in dirlist:
        stock_wei = pd.read_csv(os.path.join(panel_path, f), encoding='gbk', engine='python')

        stock_wei = stock_wei.set_index('wind_code')
        if f.split('.')[0] in indus_infor.columns:
            stock_wei['first_indus'] = indus_infor[f.split('.')[0]]
        else:
            stock_wei['first_indus'] = indus_infor[indus_infor.columns[-1]]

        stock_wei = stock_wei.dropna(axis=0, how='any')
        stock_wei['i_weight'] = 100*stock_wei['i_weight']/np.sum(stock_wei['i_weight'])

        grouped = stock_wei[['i_weight', 'first_indus']].groupby('first_indus')
        indus_wei = grouped.sum()
        indus_wei = indus_wei.T
        indus_wei.index = [f.split('.')[0]]

        indux_wei_total = pd.concat([indux_wei_total, indus_wei], axis=0)
        indux_wei_total = indux_wei_total.fillna(0)

    indux_wei_total.to_csv(os.path.join(save_path, '一级行业权重.csv'),  encoding='gbk')


def form_panel2matrix(panel_path, save_path):

    dirlist = os.listdir(panel_path)

    stock_wei_total = pd.DataFrame()
    for f in dirlist:
        # f = dirlist[0]
        stock_wei = pd.read_csv(os.path.join(panel_path, f), encoding='gbk', engine='python')
        stock_wei = stock_wei.set_index('wind_code')
        stock_wei = pd.DataFrame(stock_wei['i_weight'].values, index=stock_wei.index, columns=[f.split('.')[0]])

        stock_wei_total = pd.concat([stock_wei_total, stock_wei], axis=1)

    stock_wei_total.fillna(0)
    stock_wei_total.to_csv(os.path.join(save_path, '股票权重.csv'),  encoding='gbk')


def section_stock_infor(code_list, trade_date):
    w.start()
    if not w.isconnected():
        print('Wind连接失败，退出')
        return None

    code_str = ""
    for c in code_list:
        if len(code_str) == 0:
            code_str = c
        else:
            code_str = code_str + ',' + c

    dat = w.wsd(code_str, "concept", trade_date.strftime("%Y-%m-%d"), trade_date.strftime("%Y-%m-%d"), usedf=True)

    dat = dat[1]
    # 把涉及的概念给精简一下。
    to_del_concept = ['MSCI中盘', '标普道琼斯概念', '融资融券', '富时罗素概念', '超涨', '融资融券标的', '可转债'
        , '股票质押', '标普道琼斯概念', '珠三角', 'MSCI中国概念', 'MSCI大盘', '深圳', '高价股', '含可转债'
        , '成交主力', '机构大额卖出', '体育', '股权激励', '地方国企', '养老金概念', '长三角'
        , '万得A50', '音乐产业', '护城河', '合资企业', '员工持股', '国家队', '反关税', '出口型企业'
        , '东北振兴', '台资概念', '中非合作概念', '陆股通持续净卖出', '外资并购', '陆股通建仓', '陆股通持续净买入'
        , '机构大额买入', '预增', '精准扶贫', '中小创蓝筹', '小盘成长', '电子竞技', '大盘蓝筹', '陆股通重仓'
        , '陆股通周增仓前二十', '陆股通持续净买入', '陆股通日增仓前二十', '小盘成长', '陆股通月增仓前二十', '陆股通建仓'
        , '高盈利成长股', '大消费', '二胎政策', '周期性行业', '沪伦通', '品牌龙头', '陆股通周卖出前二十'
        , '陆股通月卖出前二十', '科技龙头', '证金概念', '陆股通周买入前二十', '机构调研', '白马股', '打板'
        , '基金重仓(季调)', '科技园区', '基金增仓', '长吉图板块', '陆股通月买入前二十', '基金减持', '一线龙头'
        , '央企']

    tmp_s = pd.Series(index=dat.index)
    for i, v in dat['CONCEPT'].items():
        try:
            c_tmp = v.split(';')
            tmp_s[i] = [c for c in c_tmp if c not in to_del_concept]
        except Exception as e:
            pass

    dat['CONCEPT'] = tmp_s
    dat.index.name = 'CODE'

    return dat


def update_stock_expect_infor():
    stocks = stocks_basis()
    code_list = list(stocks['ts_code'])
    tds = trade_days()
    res_tmp = section_stock_infor(code_list, trade_date=tds[-1])
    res_tmp.to_csv(os.path.join(r'D:\pythoncode\IndexEnhancement\barra_cne6\basic', 'concept.csv'), encoding='gbk')


def update_index_data():
    update_index_data_daily()
    update_index_data_monthly()


def update_index_data_daily():
    w.start()
    index_path = os.path.join(data_dair, 'index')
    name_map_dict = {'881001.WI': 'WindA',
                     '000300.SH': 'HS300',
                     '000016.SH': 'SZ50',
                     '000905.SH': 'ZZ500',
                     '000001.SH': '上证指数',
                     }

    if not os.path.exists(os.path.join(index_path, 'index_price_daily.csv')):
        st_dt = datetime(2006, 1, 1)
        index_price = pd.DataFrame()
    else:
        index_price = pd.read_csv(os.path.join(index_path, 'index_price_daily.csv'), engine='python')
        index_price.set_index(index_price.columns[0], inplace=True)
        index_price.index = pd.to_datetime(index_price.index)
        st_dt = index_price.index[-1] + timedelta(1)

    ed_dt = datetime.today() - timedelta(1)
    tds = trade_days()
    days_to_update = [d for d in tds if st_dt <= d <= ed_dt]
    if len(days_to_update) == 0:
        print('指数日度表现已经更新到最新日期，自动退出')
        return None

    res_df = w.wsd("881001.WI,000300.SH,000016.SH,000905.SH,000001.SH", "close", st_dt.strftime('%Y-%m-%d'),
             ed_dt.strftime('%Y-%m-%d'), usedf=True)
    res_df = res_df[1]

    res_df.rename(name_map_dict, axis=1, inplace=True)
    res_df.index.name = 'date'

    index_price = pd.concat([index_price, res_df], axis=0)
    index_price.index = pd.to_datetime(index_price.index)
    index_price.to_csv(os.path.join(index_path, 'index_price_daily.csv'), encoding='gbk')
    print('指数日度表现更新完毕，退出')


def update_index_data_monthly():
    w.start()
    index_path = os.path.join(data_dair, 'index')
    name_map_dict = {'881001.WI': 'WindA',
                     '000300.SH': 'HS300',
                     '000016.SH': 'SZ50',
                     '000905.SH': 'ZZ500',
                     }

    if not os.path.exists(os.path.join(index_path, 'index_price_monthly.csv')):
        st_dt = datetime(2006, 1, 1)
        index_price = pd.DataFrame()
    else:
        index_price = pd.read_csv(os.path.join(index_path, 'index_price_monthly.csv'), engine='python')
        index_price.set_index(index_price.columns[0], inplace=True)
        index_price.index = pd.to_datetime(index_price.index)
        st_dt = index_price.index[-1] - timedelta(90)

    ed_dt = datetime.today() - timedelta(1)
    # if st_dt.month == ed_dt.month:
    #     print('月度数据尚未更新，自动退出')
    #     return None

    res_df = w.wsd("881001.WI,000300.SH,000016.SH,000905.SH", "close", st_dt.strftime('%Y-%m-%d'),
                   ed_dt.strftime('%Y-%m-%d'), "Period=M", usedf=True)
    res_df = res_df[1]

    res_df.rename(name_map_dict, axis=1, inplace=True)
    res_df.index.name = 'date'

    to_del = [i for i in index_price.index if i > st_dt]
    if len(to_del) > 0:
        index_price.drop(to_del, axis=0, inplace=True)

    index_price = pd.concat([index_price, res_df], axis=0)
    index_price.to_csv(os.path.join(index_path, 'index_price_monthly.csv'), encoding='gbk')
    print('指数月度价格更新完毕！')


def update_macro_data():
    update_monthly_macro_data()
    update_daily_macro_data()
    print('宏观数据更新完毕----------')


def update_daily_macro_data():
    code_dict = {'S0059744': '中债国债到期收益率：1年',
                 'S0059749': '中债国债到期收益率：10年',
                 'S0059771': '中债企业债到期收益率（AAA）：1年',

                 'S5914515': '水泥价格指数：全国',
                 'S5907372': '中国玻璃综合指数',
                }
    w.start()
    str_t = ''
    for k in code_dict.keys():
        str_t = str_t + ',' + k
    str_t = str_t.lstrip(',')
    bp = r'D:\pythoncode\IndexEnhancement\barra_cne6\macro_data'
    if not os.path.exists(os.path.join(bp, 'daily_macro_data_raw.csv')):
        st_dt = datetime(2006, 1, 1)
        daily_macro = pd.DataFrame()
    else:
        daily_macro = pd.read_csv(os.path.join(bp, 'daily_macro_data_raw.csv'), engine='python')
        daily_macro.set_index(daily_macro.columns[0], inplace=True)
        daily_macro.index = pd.to_datetime(daily_macro.index)
        st_dt = daily_macro.index[-1] - timedelta(30)

    ed = datetime.today()
    res = w.edb(str_t, st_dt.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"), usedf=True)
    res = res[1]
    res.rename(code_dict, axis=1, inplace=True)

    # 为了解决月度不同数据滞后期不同的问题，把res里面有的月份，在month_macro中都删除掉。
    to_del = [i for i in res.index if i in daily_macro.index]
    daily_macro.drop(to_del, axis=0, inplace=True)
    daily_macro = pd.concat([daily_macro, res], axis=0)
    daily_macro.to_csv(os.path.join(bp, 'daily_macro_data_raw.csv'), encoding='gbk')

    print('日度宏观数据更新完毕')


def update_monthly_macro_data():
    # 因宏观数据有滞后期，所以把宏观数据分为原始数据和处理过的两类数据分别进行存储。而且不同的宏观数据滞后的月份不同，
    # 所以更新数据的时候，要一次多下载几个月的数据，避免处理不同滞后期的数据了。
    # 而且数据滞后问题，仅在回测时需要处理，不能把最近月份的数据删除了，否则策略跟踪时最近的数据都不见了。需要把数据移动的步骤
    # 迁移到测试阶段来做。

    code_dict = { # 金融数据类
                 'M0043815': '短期贷款利率:6个月至1年',
                 'M0043816': '中长期贷款利率:1至3年',
                 'M0001611': '金融机构:人民币:资金运用合计',
                 'M0001383': 'M1:同比',
                 'M0001385': 'M2:同比',
                 'M5206730': '社会融资规模:当月值',
                 # 宏观价格指数类
                 'M0017126': 'PMI',
                 'M0017131': 'PMI:产成品库存',
                 'M0000612': 'CPI当月同比',
                 'M0001227': 'PPI:全部工业品：当月同比',
                 'M0001266': 'PPI:建筑材料工业：当月同比',
                 'M0001265': 'PPI:机械工业：当月同比',
                 # 进出口类
                 'M0000607': '出口金融:当月同比',
                 # 固定资产投资与房地产投资类
                 'S0029657': '房地产开发投资完成额：累计同比',
                 'S0073290': '房屋施工面积：累计同比',
                 'S0073297': '房屋竣工面积：累计同比',
                 'S0073300': '商品房销售面积：累计同比',
                 'S0044751': '国房景气指数',
                 'M0000273': '固定资产投资完成额：累计同比',
                 'M0000275': '新增固定资产投资完成额：累计同比',
                 'M5440435': '固定资产投资完成额：基础设施建设投资：累计同比',
                 }

    w.start()

    str_t = ''

    for k in code_dict.keys():
        str_t = str_t + ',' + k
    str_t = str_t.lstrip(',')

    bp = r'D:\pythoncode\IndexEnhancement\barra_cne6\macro_data'
    if not os.path.exists(os.path.join(bp, 'month_macro_data_raw.csv')):
        st_dt = datetime(2006, 1, 1)
        month_macro = pd.DataFrame()
    else:
        month_macro = pd.read_csv(os.path.join(bp, 'month_macro_data_raw.csv'), engine='python')
        month_macro.set_index(month_macro.columns[0], inplace=True)
        month_macro.index = pd.to_datetime(month_macro.index)
        st_dt = month_macro.index[-1] - timedelta(90)

    ed = datetime.today()
    res = w.edb(str_t, st_dt.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"), usedf=True)
    res = res[1]
    res.rename(code_dict, axis=1, inplace=True)

    # 为了解决月度不同数据滞后期不同的问题，把res里面有的月份，在month_macro中都删除掉。
    to_del = [i for i in res.index if i in month_macro.index]
    month_macro.drop(to_del, axis=0, inplace=True)
    month_macro = pd.concat([month_macro, res], axis=0)
    month_macro.to_csv(os.path.join(bp, 'month_macro_data_raw.csv'), encoding='gbk')

    print('月度宏观数据更新完毕')


def update_industry_data():
    w.start()
    data = Data()
    index_path = os.path.join(data_dair, 'index')
    try:
        indus_p = data.industry_price_monthly
        st = indus_p.index[-1] - timedelta(90)
    except Exception as e:
        indus_p = pd.DataFrame()
        st = datetime(2006, 1, 1)

    ed = datetime.today() - timedelta(1)

    targets_str = ''
    for key in code_name_map_sw.keys():
        targets_str = targets_str + ',' + key

    targets_str = targets_str.lstrip(',')

    res = w.wsd(targets_str, "close", st.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"), "Period=M", usedf=True)
    res = res[1]
    res.index = pd.to_datetime(res.index)
    res = res.rename(code_name_map_sw, axis=1)

    if indus_p.empty:
        res.to_csv(os.path.join(index_path, 'industry_price_monthly.csv'), encoding='gbk')
    else:
        to_deal_index = [i for i in indus_p if i in res.index]
        indus_p.drop(to_deal_index, axis=0, inplace=True)
        indus_p = pd.concat([indus_p, res], axis=0)
        indus_p.to_csv(os.path.join(index_path, 'industry_price_monthly.csv'), encoding='gbk')

    print('行业指数更新完毕')


# w.wsd("000001.SZ", "west_netprofit_YOY", "2020-07-08", "2020-08-06", "Period=M")


def update_monthlyreports(special_year=None, update_or_not=False):
    path = os.path.join(data_dair, 'download_from_juyuan')
    data = Data()
    w.start()
    mes = generate_months_ends()
    # 仅提取特定年份的数据
    if special_year:  # = 2019
        mes0 = [m for m in mes if m.year == special_year]
        tds = mes0[0].strftime("%Y-%m-%d")
        eds = mes0[-1].strftime("%Y-%m-%d")

    try:
        tmp_df = data.monthlyreports
        tds = tmp_df.columns[-1].strftime("%Y-%m-%d")
    except Exception as e:
        tmp_df = pd.DataFrame()
        tds = datetime(2009, 1, 1).strftime("%Y-%m-%d")

    for m in mes0:
        ye = m.year
        mo = m.month
        res = w.wset("securitytradermonthlyreports", "year="+str(ye)+";month=" + '0' + str(mo)+";type=是", usedf=True)
        res = res[1]
        res['report_date'] = m.strftime("%Y-%m-%d")
        tmp_df = pd.concat([tmp_df, res], axis=0)

    data.save(tmp_df, 'monthlyreports', save_path=path)


def update_f_data_from_wind(special_year=None, update_or_not=True):
    # special_year是在提取历史数据时，因Wind提取数据受限原因导致，可能一次仅提取几年的数据

    path = os.path.join(data_dair, 'download_from_juyuan')
    w.start()
    data = Data()
    stock_basic_inform = data.stock_basic_inform

    mes = generate_months_ends()
    calendar_ends = generate_calendar_month_ends()

    iterms = [StructWind('rd_exp', 'Q', 'unit=1;rptType=1;Days=Alldays'),
              StructWind('west_netprofit_YOY', 'M'),
              # StructWind('stmnote_mgmt_ben_top3m', 'Y', "unit=1;Days=Weekdays", 'year=', 2007),
              StructWind('est_instnum', 'M', None, 'year=', 2021),
              # est_instnum为预测机构家数数据，参数为预测的年份，需注意的是日期的年份应该小于参数的年份1年，比如提取19年的数据时，
              # 参数应该是2020，提取18年数据的时候，参数应该是2019.
              # 带年份参数的数据，因每次提取前需要设定年份参数，一次只能提取一年的。
             ]

    codes_str = ''
    for i in stock_basic_inform.index:
        codes_str = codes_str + ',' + i
    codes_str = codes_str[1:]

    eds = datetime.today().strftime("%Y-%m-%d")

    for it in iterms:
        name = it.name
        period = it.period
        other = it.get_str()
        try:
            tmp_df = eval('data.'+name.lower())
            tmp_df.columns = pd.to_datetime(tmp_df.columns)
            tds = tmp_df.columns[-1]
        except Exception as e:
            tmp_df = pd.DataFrame()
            tds = datetime(2009, 1, 1)

        # 删掉不是月末数据的
        # todel = [col for col in tmp_df.columns if col not in mes]
        # if len(todel) > 0:
        #     tmp_df

        # 仅提取特定年份的数据
        if special_year:     # = 2019
            mes0 = [m for m in mes if m.year == special_year]
            tds = mes0[0].strftime("%Y-%m-%d")
            eds = mes0[-1].strftime("%Y-%m-%d")

        if update_or_not and not tmp_df.empty:
            m_to_update = [m for m in mes if m not in tmp_df.columns and m > tmp_df.columns[-1]]
            if len(m_to_update) == 0:
                print('{}无需更新'.format(name))
                continue

        # 该变量下日期无效，开始和结束日期随便设一个同一年的就好。
        # if name == 'stmnote_mgmt_ben_top3m':
        #     tds = mes[-2].strftime("%Y-%m-%d")
        #     eds = mes[-1].strftime("%Y-%m-%d")

        if not special_year and period == 'Q' and (datetime.today() - tds).days < 110:
            print('{}无需更新'.format(name))
            continue
        elif not special_year and period == 'M' and (datetime.today() - tds).days < 20:
            print('{}无需更新'.format(name))
            continue

        res_tmp = w.wsd(codes_str, name, tds, eds, other,
                        usedf=True)
        res_tmp1 = res_tmp[1]
        res_tmp1 = res_tmp1.T

        tmp_df = pd.concat([tmp_df, res_tmp1], axis=1)
        # 读取本地数据时和Wind提取数据时的时间格式可能不一样，统一一下才能排序
        tmp_df.columns = pd.to_datetime(tmp_df.columns)
        # 把columns排序
        tt = list(tmp_df.columns)
        tt.sort()
        tmp_df = tmp_df[tt]

        # 最后一列可能不是月末日期，需要删除
        if tmp_df.columns[-1] not in calendar_ends and tmp_df.columns[-1] not in mes:
            tmp_df.drop(tmp_df.columns[-1], axis=1, inplace=True)

        if tmp_df[tmp_df.columns[-1]].isna().sum() == len(tmp_df.index):
            tmp_df.drop(tmp_df.columns[-1], axis=1, inplace=True)

        data.save(tmp_df, name.lower(), save_path=path)


def update_index_wei():
    w.start()
    data = Data()
    zz500_wt = data.zz500_wt
    hs300_wt = data.hs300_wt

    mes = generate_months_ends()
    # 先删除一些不是月末的数据
    to_del = [c for c in zz500_wt.columns if c not in mes]
    if len(to_del) > 0:
        zz500_wt = zz500_wt.drop(to_del, axis=1)
    to_del = [c for c in hs300_wt.columns if c not in mes]
    if len(to_del) > 0:
        hs300_wt = hs300_wt.drop(to_del, axis=1)

    new_mes = [m for m in mes if m > zz500_wt.columns[-1]]

    for m in new_mes:
        m_str = m.strftime("%Y-%m-%d")
        # 沪深300
        res = w.wset("indexconstituent", "date=" + m_str + ";windcode=000300.SH", usedf=True)
        res = res[1]
        res.set_index('wind_code', inplace=True)
        to_add = pd.DataFrame({m: res['i_weight']})
        hs300_wt = pd.concat([hs300_wt, to_add], axis=1)

        # 中证500
        res = w.wset("indexconstituent", "date=" + m_str + ";windcode=000905.SH", usedf=True)
        res = res[1]
        res.set_index('wind_code', inplace=True)
        to_add = pd.DataFrame({m: res['i_weight']})
        zz500_wt = pd.concat([zz500_wt, to_add], axis=1)

    data.save(hs300_wt, 'hs300_wt', save_path=os.path.join(data_dair, 'index'))
    data.save(zz500_wt, 'zz500_wt', save_path=os.path.join(data_dair, 'index'))
    print('指数权重更新完毕')


# 更新行业基本面高频数据，如部分价格、库存、等数据
def update_industry_basic_data():
    pass


def fund_indus_wei():
    data = Data()

    mes = generate_months_ends()
    stock_basic_inform = data.stock_basic_inform

    # 读取流通市值数据
    negomv = data.negotiablemv_daily

    # 读取基金重仓股数据
    db = shelve.open(r'D:\Database_Stock\Data\fund\fund_portfolio.db')
    fund_dict = db['fund_portfolio_float_rate']
    db.close()

    def date_compare(target, d_list):
        res = None
        for d in d_list:
            if target.year == d.year and target.month == d.month:
                res = d
                break
        return res

    # 根据这些基金的持仓，计算持有股票的市值比例
    ratio_df = pd.DataFrame(0, index=stock_basic_inform.index, columns=mes)
    for key, hold_df in fund_dict.items():
        for col in hold_df.columns:

            # 选择出对应的月份
            finded = 0
            for all_col in ratio_df.columns:
                if all_col.year == col.year and all_col.month == col.month:
                    finded = all_col
                    break

            # 太早的数据，如02年的
            if finded == 0:
                continue

            # 一个小bug
            if col.month not in [1, 4, 7, 10]:
                continue

            target_month = ratio_df[finded]

            tmp_se = hold_df[col].dropna()
            # 针对对应的股票，市值相加
            for stock in tmp_se.index:
                if stock in target_month.index:
                    target_month[stock] = target_month[stock] + tmp_se[stock]

            # target_month.sum()
            ratio_df[finded] = target_month

    # 对于空余的月份，直接复制前值
    for col_n in range(1, len(ratio_df.columns)):
        if ratio_df[ratio_df.columns[col_n]].sum() == 0 and \
                ratio_df[ratio_df.columns[col_n - 1]].sum() != 0:
            ratio_df[ratio_df.columns[col_n]] = ratio_df[ratio_df.columns[col_n - 1]]

    ratio_df = ratio_df / 100

    new_cols = [col for col in negomv.columns if col in mes]
    negomv = negomv[new_cols]
    topten_to_float_ashare = ratio_df * negomv

    topten = topten_to_float_ashare.dropna(how='all', axis=1)

    stock_inform = data.stock_basic_inform

    topten = topten.drop(topten.columns[topten.sum() == 0], axis=1)

    indus_sw = stock_inform['申万一级行业']

    indus_wei = pd.DataFrame()
    for col, se in topten.iteritems():
        tmp_se = pd.concat([se, indus_sw], axis=1)
        grouped = tmp_se.groupby(tmp_se.columns[1])
        tmp_dict = {}
        for name, wei_tmp in grouped:
            tmp_dict.update({name: wei_tmp[wei_tmp.columns[0]].sum()})

        se_tmp = pd.Series(tmp_dict)
        se_tmp = se_tmp / se_tmp.sum()
        se_df = pd.DataFrame({col: se_tmp})
        indus_wei = pd.concat([indus_wei, se_df], axis=1)

    indus_wei_ma = ma(indus_wei, 3)
    indus_wei_ma.to_csv(r'D:\Database_Stock\Data\fund\indus_wei_ma.csv', encoding='gbk')

    return indus_wei_ma


# 下载盈利预告数据
def down_profitnotice(save_path=None, update=True):

    if not save_path:
        save_path = r'D:\Database_Stock\Data\profitnotice'

    if not update:
        hasd_f_list = []
    else:
        hasd_f_list = os.listdir(save_path)

    w.start()
    data = Data()
    basic_inform = data.stock_basic_inform

    code_str = ''
    for i in basic_inform.index:
        code_str = code_str + ','
        code_str = code_str + i

    code_str = code_str[1:]

    mes = generate_months_ends()
    # 把交易日月末日期转换成日历日月末日期
    calendar_mes = []
    for m in mes:
        d = calendar.monthrange(m.year, m.month)[1]
        calendar_mes.append(date(m.year, m.month, d))

    wanted_mes = [m for m in calendar_mes if m.month in [3, 6, 9, 12] and m.year > 2014
                  and m.strftime("%Y-%m-%d")+'.csv' not in hasd_f_list]

    for m in wanted_mes:
        m_str = "rptDate=" + m.strftime("%Y%m%d")
        res = w.wss(code_str,
                    "profitnotice_changemax,profitnotice_changemin,profitnotice_firstdate,profitnotice_style",
                     m_str, usedf=True)
        res = res[1]
        res.dropna(how='any', axis=0, inplace=True)
        res.index.name = 'Code'
        res.to_csv(os.path.join(save_path, m.strftime("%Y-%m-%d")+'.csv'), encoding='gbk')


# 处理业绩预告数据
# 处理逻辑：首先对公告日期修正，修正为当月末的日期，然后检查相应月末的一致预期数据，选择一期预期调升的。
# 然后先统计一下满足上面条件的个股的数量

def profitnotice_process(data_path=None, save_path=None):

    if not data_path:
        data_path = r'D:\Database_Stock\Data\profitnotice'

    # 整理成真实月末日期的dict形式，披露时间月末日为key，相应的df为value
    announce_dict = defaultdict(pd.DataFrame)
    file_list = os.listdir(data_path)
    for f in file_list:
        dat = pd.read_csv(os.path.join(data_path, f), encoding='gbk')
        dat.set_index(dat.columns[0], inplace=True)
        stmp_tmp = f.split('.')[0]
        stmp_dt = datetime.strptime(stmp_tmp, "%Y-%m-%d")

        # 仅保留业绩预增的个股
        dat = dat.loc[dat.index[dat['PROFITNOTICE_STYLE'] == '预增'], :]

        mes = generate_months_ends()

        def tmp_fun(mm, mes):
            for m in mes:
                if mm.year == m.year and mm.month == m.month:
                    return m

        # 逐行遍历，调整为正确的日期
        for code, se in dat.iterrows():
            ana_date = pd.to_datetime(se['PROFITNOTICE_FIRSTDATE'])
            adjust_date = tmp_fun(ana_date, mes)
            df = announce_dict[adjust_date]
            df = pd.concat([df, pd.DataFrame({code: se}).T], axis=0)
            announce_dict[adjust_date] = df

    # 整理成key为交易日月末日期，vale为df的dict，看一下数量
    for key, value in announce_dict.items():
        print(key)
        print(len(value))

    # 再使用一致预期数据过滤，选择一致预期数据月末相对上月调升的标的


if __name__ == '__main__':

    update_f_data_from_wind()

    # update_monthlyreports()
    # profitnotice_process()
    # fund_indus_wei()
    # update_f_data_from_wind()

    # update_stock_basic_inform()
    # update_stock_expect_infor()

    # save_path = r'D:\pythoncode\IndexEnhancement\上证综指权重'
    # panel_path = os.path.join(save_path, '月度数据(股票)')
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # if not os.path.exists(panel_path):
    #     os.mkdir(panel_path)
    #
    # get_composi_and_wei(panel_path)
    # form_stock2_first_indus(panel_path, save_path)
    # form_stock2_second_indus(panel_path, save_path)
    # form_panel2matrix(panel_path, save_path)


