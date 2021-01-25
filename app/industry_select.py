from datetime import datetime
import pandas as pd
import numpy as np
from utility.tool0 import Data
from sklearn import preprocessing
from utility.factor_data_preprocess import info_cols, add_to_panels, apply_func2, align,\
                                           concat_factor_2, simple_func, drop_some, winsorize
from utility.stock_pool import compute_icir
import os
from utility.constant import root_dair, industry_factor_dict

'''
基于多因子打分法的行业轮动模型，针对申万三级行业。
行业层面有效因子较少，主要使用动量、盈利、成长三大类因子。
最后的结果需要更多的结合行业基本面逻辑进行选择判断，定性分析因素占比远大于定量因素。
具体因子可见 constant.py 中定义的常量 industry_factor_names
原始报告可见《行业多因子轮动模型――金融工程专题报告》。
'''


# MAD:中位数去极值
def filter_extreme_mad(series, n):
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    return np.clip(series, min_range, max_range)


class IndustrySelect:
    def __init__(self, alpha_factor_dict=industry_factor_dict, update_only=True):
        self._data = Data()
        self.stock_factor_path = os.path.join(root_dair, '因子预处理模块', '因子')
        self.industry_level = None
        self.industry_factor_path = None
        self.save_path = None
        self.stock_basic_inform = None
        self.factors_dict = alpha_factor_dict
        self.update_only = update_only
        self.max_num = None
        self.indus_selected = None
        self.indus_rank = None

    def set_industry_level(self, level=3):
        self.industry_level = level
        if level == 3:
            self.industry_factor_path = os.path.join(root_dair, '行业多因子', '申万三级', '因子')
            self.save_path = os.path.join(root_dair, '行业多因子', '申万三级')
            self.max_num = 20

        elif level == 1:
            self.industry_factor_path = os.path.join(root_dair, '行业多因子', '申万一级', '因子')
            self.save_path = os.path.join(root_dair, '行业多因子', '申万一级')
            self.max_num = 3

    # 定义不同因子的合成方式，财务因子使用中位数法合成，价量因子使用市值加权法合成
    def compose_way(self):
        median_factors_0 = ['Sales_G_q', 'Profit_G_q', 'ROE_G_q', 'ROE_q', 'ROE_ttm', 'ROA_q', 'ROA_ttm',
                            'Grossprofitmargin_q',
                            'Grossprofitmargin_ttm', 'Profitmargin_q', 'Profitmargin_ttm', 'Assetturnover_q',
                            'Assetturnover_ttm', 'Operationcashflowratio_q', 'Operationcashflowratio_ttm',
                            'Sue', 'Revsu']

        median_factors_1 = []
        # 改成首字母大写后续字母小写的格式
        for col in median_factors_0:
            new_c = col[0].upper() + col[1:].lower()
            median_factors_1.append(new_c)

        compose_way = {'median': median_factors_1}
        self.compute_factor(compose_way)

    # 根据不同因子的合成方式，计算行业因子，并保存到指定文件夹下
    def compute_factor(self, compose_way):
        data_path = self.stock_factor_path
        indus_save_path = self.industry_factor_path

        if self.industry_level == 3:
            industry_str = '申万三级行业'
        elif self.industry_level == 1:
            industry_str = '申万一级行业'

        self.stock_basic_inform = self._data.stock_basic_inform
        # 创建文件夹
        if not os.path.exists(os.path.join(indus_save_path)):
            os.makedirs(os.path.join(indus_save_path))

        if not self.update_only:
            to_process_f = os.listdir(data_path)
        elif self.update_only:
            fls = os.listdir(data_path)
            processed_list = os.listdir(indus_save_path)
            to_process_f = [f for f in fls if f not in processed_list]

        if len(to_process_f) == 0:
            print('无需要处理的数据')
            return None

        for panel_f in to_process_f:
            print(panel_f)
            # panel_f = os.listdir(date_path)[0]
            panel_dat = pd.read_csv(os.path.join(data_path, panel_f),
                                    encoding='gbk', engine='python',
                                    index_col=['Code'])

            # 需要先对股票因子做两个常规处理
            data_to_process = drop_some(panel_dat)
            # data_to_process.empty
            data_to_process = winsorize(data_to_process)
            try:
                data_to_process = pd.concat([data_to_process, self.stock_basic_inform[industry_str]], axis=1, join='inner')
            except Exception as e:
                print('debug')
            factors_to_concat = list((set(panel_dat.columns) - (set(info_cols) - set(['Pct_chg_nm']))))
            grouped = data_to_process.groupby(industry_str)

            ind_factor = pd.DataFrame()
            for name, group in grouped:
                factor_dat = group[factors_to_concat]
                mv = group['Mkt_cap_float']
                factor_dat = factor_dat.applymap(apply_func2)
                factor_concated = {}
                for factor_name, factors in factor_dat.iteritems():
                    if factor_name == 'Lncap_barra':
                        tmp_f = np.log(np.sum(group['Mkt_cap_float']))
                        factor_concated.update({factor_name: tmp_f})
                        continue

                    # 不同类型因子有不同的合成方式
                    factor_concat_way = 'mv_weighted'
                    for concat_way, factorlist in compose_way.items():
                        factorlist_tmp = [fa.lower() for fa in factorlist]
                        if factor_name.lower() in factorlist_tmp:
                            factor_concat_way = concat_way
                    tmp_f = simple_func(factors, mv=group['Mkt_cap_float'], type=factor_concat_way)

                    factor_concated.update({factor_name: tmp_f})

                factor_concated = pd.DataFrame(factor_concated)
                factor_concated.index = [name]
                factor_concated.loc[name, 'Mkt_cap_float'] = np.sum(mv)  # 市值采用行业市值和
                if 'Industry_zx' in group.columns:
                    factor_concated.loc[name, 'Industry_zx'] = group.loc[group.index[0], 'Industry_zx']
                if 'Industry_sw' in group.columns:
                    factor_concated.loc[name, 'Industry_sw'] = group.loc[group.index[0], 'Industry_sw']
                ind_factor = pd.concat([ind_factor, factor_concated], axis=0)

            ind_factor.index.name = 'Name'
            ind_factor.to_csv(os.path.join(indus_save_path, panel_f), encoding='gbk')

    # 等权加成
    def select_indus(self):
        max_num = self.max_num

        if self.industry_level == 3:
            industry_str = '申万三级行业'
        elif self.industry_level == 1:
            industry_str = '申万一级行业'

        # 选出的几个头部行业
        indus_selected = pd.DataFrame()
        # 所有行业的排序
        indus_value_rank = pd.DataFrame()

        panel_list = os.listdir(self.industry_factor_path)
        for fn in panel_list:

            f_datetime = datetime.strptime(fn.split('.')[0], "%Y-%m-%d")

            data = pd.read_csv(os.path.join(self.industry_factor_path, fn), engine='python', encoding='gbk')
            data.set_index('Name', inplace=True)

            data_concat = pd.DataFrame()
            for key, value in self.factors_dict.items():
                vs = [v for v in value if v in data.columns]
                if len(vs) == 0:
                    continue

                data_fed = data[vs].apply(filter_extreme_mad, args=(3,))   # 去极值
                data_fed = data_fed.replace([np.inf, -np.inf], np.nan)
                mi = data_fed.min().min()
                data_fed = data_fed.replace(np.nan, mi)

                data_scaled_v = preprocessing.scale(data_fed)              # 按列标准化

                data_scaled = pd.DataFrame(data=data_scaled_v, index=data_fed.index, columns=data_fed.columns)
                tmp_se = data_scaled.mean(axis=1)
                tmp_df = pd.DataFrame({key: tmp_se})
                data_concat = pd.concat([data_concat, tmp_df], axis=1)

            len(data.index)

            wei = np.ones(len(data_concat.columns))/len(data_concat.columns)
            val = data_concat.values
            scores = pd.Series(index=data.index, data=np.dot(val, wei))

            data_concat['score'] = scores
            data_concat = data_concat.sort_values('score', ascending=False)

            if fn == panel_list[-1]:
                print('保存最后一期结果做进一步分析')
                f_to_t = self.stock_basic_inform[['申万一级行业', '申万三级行业']]
                f_to_t = f_to_t.set_index('申万三级行业')
                f_to_t = f_to_t[~f_to_t.index.duplicated(keep='first')]
                data_concat['申万一级行业'] = f_to_t
                data_concat = data_concat.round(decimals=3)                 # 保留三位小数
                data_concat.to_csv(os.path.join(self.save_path, '最后一期详细结果.csv'), encoding='gbk')

            scores_sorted = scores.sort_values(ascending=False)

            # 记录选出的头部行业
            selected_indus = list(scores_sorted.index[:max_num])
            selected_df = pd.DataFrame(index=range(0, len(selected_indus)), columns=[f_datetime], data=selected_indus)
            indus_selected = pd.concat([indus_selected, selected_df], axis=1)

            # 记录排序信息
            rank_tmp = pd.DataFrame(index=list(scores_sorted.index), columns=[f_datetime],
                                    data=list(range(1, len(scores_sorted)+1)))
            indus_value_rank = pd.concat([indus_value_rank, rank_tmp], axis=1)

        indus_selected.to_csv(os.path.join(self.save_path, '行业选择结果.csv'), encoding='gbk')
        indus_value_rank.to_csv(os.path.join(self.save_path, '行业排序结果.csv'), encoding='gbk')

        self.indus_selected = indus_selected
        self.indus_rank = indus_value_rank

    def show_newest(self):
        print(self.indus_selected[self.indus_selected.columns[-1]])
        print(self.indus_rank[self.indus_rank.columns[-1]])


# 确定行业权重
def generate_indus_wei(rate_1_5=0.1, rate_6_10=0.06, rate_11_15=0.04, rate_over_15=0,
                       coef_rank=0.5, coef_fund_top=0.5):
    '''
    目的：使用基金重仓股的行业权重和行业轮动模型的行业排序信息合成行业权重，跳出基于宽基指数做行业配置。
    逻辑：根据行业轮动模型的排序信息，排名1-5的行业配置10%的权重，6-10的行业配置5%的权重，11-20的行业配置2.5%的权重。
          上面的权重再加上基金重仓股的权重，再做一个权重和为1的运算处理。

    如果仅投资申万行业指数的话，只要上面的组合收益跑赢宽基指数就证明有效了

    rate_1_5, rate_6_10, rate_11_15, rate_over_15 对排序在相应区间内的权重设置，
    coef_rank=0.5, coef_fund_top=0.5 排序权重和基金重仓股的权重

    :return:
    '''

    path = r'D:\Database_Stock\行业多因子\申万一级\行业排序结果.csv'
    rank_infor = pd.read_csv(path, encoding='gbk')
    rank_infor.set_index(rank_infor.columns[0], inplace=True)

    p = r'D:\Database_Stock\Data\fund\indus_wei_ma.csv'
    fund_top10_wei = pd.read_csv(p, encoding='gbk')
    fund_top10_wei = fund_top10_wei.set_index(fund_top10_wei.columns[0])
    fund_top10_wei.columns = pd.to_datetime(fund_top10_wei.columns)
    fund_top10_wei = fund_top10_wei.dropna(how='all', axis=1)

    fund_top10_wei.sum()

    # 排名1-5的行业配置10%的权重，6-10的行业配置6%的权重，11-15的行业配置2.5% 的权重
    for k in range(1, 6, 1):
        rank_infor.replace({k: rate_1_5}, inplace=True)
    for k in range(6, 11, 1):
        rank_infor.replace({k: rate_6_10}, inplace=True)
    for k in range(11, 16, 1):
        rank_infor.replace({k: rate_11_15}, inplace=True)
    for k in range(16, 29, 1):
        rank_infor.replace({k: rate_over_15}, inplace=True)

    total_wei = coef_fund_top*fund_top10_wei + coef_rank*rank_infor
    total_wei.dropna(how='all', axis=1, inplace=True)
    res_wei = total_wei / total_wei.sum()

    res_wei.index.name = '申万行业'

    # 一个简单回测，权重乘以行业指数的收益率
    data = Data()
    industry_price_monthly = data.industry_price_monthly
    industry_rate = industry_price_monthly / industry_price_monthly.shift(axis=1)
    industry_rate = industry_rate - 1
    industry_rate.dropna(how='all', axis=1, inplace=True)

    rate_np = industry_rate.shift(axis=1)

    # 更改名称，去掉（申万）
    new_index = [i.split('（')[0] for i in rate_np.index]
    rate_np.index = new_index

    rate_np, wei = align(rate_np, res_wei)

    ret_df = rate_np * wei
    net_pct = ret_df.sum()
    net_value = (net_pct + 1).cumprod()

    return res_wei, net_value


if __name__ == '__main__':
    indus_wei, net = generate_indus_wei()
    indus_wei.to_csv(r'D:\Database_Stock\indus_wei.csv', encoding='gbk')

    # ise = IndustrySelect()
    # # ise.set_industry_level(level=3)
    # # ise.compose_way()
    # # ise.select_indus()
    # # ise.show_newest()
    # ise.set_industry_level(level=1)
    # ise.compose_way()
    # ise.select_indus()
    # ise.show_newest()












