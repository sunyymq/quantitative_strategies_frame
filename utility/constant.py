import os
import re
from copy import deepcopy

root_dair = r'D:\Database_Stock'
data_dair = r'D:\Database_Stock\Data'

industry_benchmark = 'sw'

# 用来选择行业的因子
industry_factor_dict = {'mom': ['Return_6m', 'Return_12m'],
                        'growth': ['Profit_g_q', 'Roe_g_q', 'Sales_g_q', 'Sue'],
                        'quality': ['Roa_ttm', 'Roe_q', 'Profitmargin_q'],
                        }

# # 不同行业用来打分的大类因子
# factor_dict_for_scores = {
#           # '建筑材料': ['quality', 'growth', 'grossrate', 'west'],
#           'default': ['quality', 'growth'],  #, 'west'],
#           }

# 默认的大类因子及其二级因子
default_dict = {'quality': ['Roa_q', 'Roe_q'],
                'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q', 'Sue'],
                }

# 不同行业的需要合成的大类因子，在默认分类的基础上，通过对默认分类的添加和修改，来得到不同行业的大类合成因子
factor_dict_to_concate = {
               'default': default_dict,
               '银行':    {'value': ['Ep', 'Bp'],
                           'growth': ['Profit_g_q', 'Sales_g_q', 'Sue'],
                          },
               '电气设备': {'quality': ['Roa_q', 'Roe_q'],
                            'value': ['Ep', 'Bp'],
                            'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q', 'Sue'],
                            'liquidity': ['Stom_barra', 'Stoq_barra'],
                          },
               '汽车': {'growth': ['Sales_g_q', 'Profit_g_q'],
                        'quality': ['Roa_q', 'Roe_q'],
                        'margin': ['Grossprofitmargin_diff'],
                        'size': ['Midcap_barra'],
                      },
               '食品饮料': {
                        'margin': ['Grossprofitmargin_q'],
                        'quality': ['Roa_q'],
                        'growth': ['Profit_g_q', 'Roe_g_q', 'Sue', 'Sales_G_q'],
                        # 'west': ['West_netprofit_yoy', 'Est_instnum'],
                        'size': ['Lncap_barra'],
                       },
               '医药生物': {
                        'margin': ['Grossprofitmargin_diff'],
                        'quality': ['Roa_q', 'profitmargin_q'],
                        'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q', 'Sue'],
                        'rd': ['Rdtosales'],
                        # 'size': ['Lncap_barra'],
                        },
               '家用电器': {
                        'value': ['BP'],
                        'quality': ['Roa_q'],
                        'margin': ['Assetturnover_q'],
                        'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q'],
                        'size': ['Lncap_barra'],
                        },
               '电子': {
                        'value': ['Peg'],
                        'margin': ['Grossprofitmargin_diff'],
                        'quality': ['profitmargin_q'],
                        'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q', 'Revsu'],
                },
               '通信': {
                        'value': ['Bp'],
                        'margin': ['Assetturnover_q'],
                        'quality': ['Roa_q', 'Roe_q'],
                        'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q', 'Sue'],
                },
               '机械设备': {
                        'value': ['Bp'],
                        'size': ['Midcap_barra'],
                        'growth': ['Revsu', 'Profit_g_q', 'Sales_g_q', 'Sue'],
                },
               '计算机': {
                        'quality': ['Grossprofitmargin_q'],
                        'growth': ['Sales_g_q', 'Profit_g_q', 'Sue', 'Roe_g_q'],
                        'west': ['West_netprofit_yoy', 'Est_instnum'],
                        'size': ['Midcap_barra'],
                        'rd': ['Rdtosales'],
                       },
               '传媒': {
                        'value': ['Peg_3'],
                        'quality': ['Assetturnover_q', 'Operationcashflowratio_q'],
                        'growth': ['Ocf_g_q', 'Sales_g_q'],
                       },
               '国防军工':
                       {'value': ['Bp'],
                        'quality': ['Grossprofitmargin_diff'],
                        'growth': ['Profit_G_q'],
                       },
               '证券Ⅱ':
                      {'value': ['Bp', 'Ep'],
                      },
               '交通运输':
                      {'growth': ['Sue'],
                       # 'size': ['Lncap_barra'],
                      },
               '房地产':
                      {'value': ['Bp'],
                      },
               '商业贸易':
                      {'value': ['Bp'],
                       'size': ['Midcap_barra'],
                       'margin': ['Grossprofitmargin_diff'],
                       'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q', 'Sue'],
                      },
               '化工':
                      {'value': ['Ep', 'Peg'],
                       'growth': ['Profit_g_q', 'Roe_g_q', 'Sales_g_q', 'Sue'],
                       'quality': ['Roa_q', 'Roe_q', 'Profitmargin_q'],
                       'rd': ['Rdtosales'],
                       'margin': ['Grossprofitmargin_diff', 'Assetturnover_q'],
                      },
               '轻工制造':
                       {
                        'margin': ['Grossprofitmargin_diff'],
                        'value': ['Ep'],
                        'quality': ['Roa_q', 'Roe_q'],
                        'size': ['Midcap_barra'],
                        'growth': ['Profit_g_q', 'Sales_g_q', 'Roe_g_q'],
                       },
               '纺织服装':
                       {'value': ['Bp'],
                        'size': ['Midcap_barra'],
                        'growth': ['Profit_g_q'],
                        'margin': ['Grossprofitmargin_diff'],
                       },
               '休闲服务':
                       {'value': ['Bp', 'Ep'],
                        'size': ['Midcap_barra'],
                        'quality': ['Roa_q', 'profitmargin_q'],
                        'growth': ['Profit_g_q', 'Sales_g_q'],
                       },
               '农林牧渔':
                       {'value': ['Bp'],
                        'margin': ['Grossprofitmargin_diff', 'Assetturnover_q'],
                       },
               '建筑装饰':
                       {'value': ['Peg'],
                        'growth': ['Roe_g_q', 'Sue', 'Revsu'],
                       },
               '建筑材料':
                       {
                        'margin': ['Grossprofitmargin_diff', 'Assetturnover_q'],
                        'growth': ['Profit_g_q', 'Roe_g_q', 'Sales_g_q', 'Sue'],
                        'quality': ['Roa_q', 'Roe_q'],
                       },
               '公用事业':
                       {'value': ['Bp'],
                        'margin': ['Grossprofitmargin_diff', 'Assetturnover_q'],
                        'growth': ['Profit_g_q', 'Sales_g_q'],
                       },
               '有色金属':
                       {'value': ['Bp'],
                        'margin': ['Grossprofitmargin_diff'],
                        'growth': ['Profit_g_q', 'Roe_g_q', 'Sales_g_q', 'Sue'],
                        'quality': ['Roa_q', 'Roe_q'],
                        # 'size': ['Lncap_barra'],
                        },
               '钢铁':
                       {
                        'growth': ['Profit_g_q', 'Roe_g_q', 'Sales_g_q', 'Revsu'],
                        'quality': ['Roa_q', 'Roe_q'],
                        'margin': ['Cashratio', 'Assetturnover_q'],
                       },
               '采掘':
                       {
                        'margin': ['Grossprofitmargin_diff'],
                        'growth': ['Profit_g_q', 'Roe_g_q', 'Sales_g_q', 'Revsu'],
                        'quality': ['profitmargin'],
                        },
                }

added_dict = {
               # '建筑材料': {'grossrate': ['Grossprofitmargin_diff', 'Totalassetturnover']}
             }

for key, value in added_dict.items():
    tmp_d = deepcopy(default_dict)
    for k, v in value.items():
        tmp_d.update({k: v})
    factor_dict_to_concate.update({key: tmp_d})


# 工作目录，存放代码和因子基本信息
work_dir = os.path.dirname(os.path.dirname(__file__))
# 经过预处理后的因子截面数据存放目录
factor_path = os.path.join(work_dir, '因子预处理模块', '因子（已预处理）')
# factor_path = os.path.join(work_dir, '行业多因子', 'second_industry', '因子(已预处理)')
# 测试结果图表存放目录
sf_test_save_path = os.path.join(work_dir, '单因子检验')
# sf_test_save_path = os.path.join(work_dir, '行业多因子', 'second_industry', '单因子检验')

# 行业多因子模块
industry_factor_path = os.path.join(work_dir, '板块多因子')


# # 测试结果图表存放目录（如无则自动生成）
# sf_test_save_path = os.path.join(work_dir, '单因子检验')
# # 行业多因子模块
# industry_factor_path = os.path.join(work_dir, '板块多因子')

factor_matrix_path = os.path.join(work_dir, '单因子检验', '因子矩阵')
# factor_matrix_path = os.path.join(work_dir, '行业多因子', 'second_industry', '单因子检验', '因子矩阵')

# 合成、正交因子存放目录（如无则自动生成）
rm_save_path = os.path.join(work_dir, '行业多因子', 'second_industry', '收益模型')
# 测试结果图表存放目录（如无则自动生成）
index_enhance_dir = os.path.join(work_dir, '指数增强模型')

select_indus_dir = os.path.join(work_dir, '行业多因子', 'second_industry', '行业选择')

# 单因子检验汇总
# total_result_path = os.path.join(work_dir, '行业多因子', 'second_industry', '结果汇总比较')
total_result_path = os.path.join(work_dir, '单因子检验', '结果汇总比较')
if not os.path.exists(total_result_path):
    os.makedirs(total_result_path)


tds_interface = 'tushare'  # 'Wind' or 'tushare'

old_info_cols = ['No', 'code', 'name', 'Name', 'ipo_date', 'industry_zx', 'industry_zz', 'mkt', 'index',
                 'industry_sw', 'MKT_CAP_FLOAT', '中信三级行业', '中信一级行业', '中信二级行业', 'Lncap_barra',
                 'is_open1', 'PCT_CHG_NM', 'second_industry', 'third_industry', 'name', 'plate',
                 ]


zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查中文
new_cols = []
for col in old_info_cols:

    if col == 'name':
        new_cols.append('Sec_name')
        continue

    match = zhmodel.search(col)
    if not match:   # 不包含中文
        new_c = col[0].upper() + col[1:].lower()
        new_cols.append(new_c)

info_cols = list(set(old_info_cols) | set(new_cols))

# 无需做因子预处理中性化的因子
non_processed_factors = ['Rps', "Topten_to_float_ashare", "Delta_to_float_ashare", 'Long_mom']

copy_cols = ['PCT_CHG_NM.csv', 'second_industry.csv', 'industry_sw.csv', 'MKT_CAP_FLOAT.csv']

# 根据中信的上下游产业链板块指数划分
plate_to_indus = {'上游资源': ['石油石化', '煤炭', '有色金属'],
                  '中游制造': ['钢铁', '基础化工', '轻工制造', '机械', '电力设备及新能源', '国防军工'],
                  '下游消费': ['汽车', '商贸零售', '消费者服务', '家电', '纺织服装', '医药', '食品饮料', '农林牧渔'],
                  '基建与运营': ['电力及公用事业', '建筑', '建材', '交通运输'],
                  'TMT': ['电子', '计算机', '通信', '传媒'],
                  '金融地产': ['银行', '非银行金融', '房地产', '综合金融'],
                 }

code_name_map_citic = {"CI005001.WI": "石油石化（中信）",
                       "CI005002.WI": "煤炭（中信）",
                       "CI005003.WI": "有色金属（中信）",
                       "CI005004.WI": "电力及公用事业（中信）",
                       "CI005005.WI": "钢铁（中信）",
                       "CI005006.WI": "基础化工（中信）",
                       "CI005007.WI": "建筑（中信）",
                       "CI005008.WI": "建材（中信）",
                       "CI005009.WI": "轻工制造（中信）",
                       "CI005010.WI": "机械（中信）",
                       "CI005011.WI": "电力设备及新能源（中信）",
                       "CI005012.WI": "国防军工（中信）",
                       "CI005013.WI": "汽车（中信）",
                       "CI005014.WI": "商贸零售（中信）",
                       "CI005015.WI": "消费者服务（中信）",
                       "CI005016.WI": "家电（中信）",
                       "CI005017.WI": "纺织服装（中信）",
                       "CI005018.WI": "医药（中信）",
                       "CI005019.WI": "食品饮料（中信）",
                       "CI005020.WI": "农林牧渔（中信）",
                       "CI005021.WI": "银行（中信）",
                       "CI005022.WI": "非银行金融（中信）",
                       "CI005023.WI": "房地产（中信）",
                       "CI005024.WI": "交通运输（中信）",
                       "CI005025.WI": "电子（中信）",
                       "CI005026.WI": "通信（中信）",
                       "CI005027.WI": "计算机（中信）",
                       "CI005028.WI": "传媒（中信）",
                       "CI005029.WI": "综合（中信）",
                       "CI005030.WI": "综合金融（中信）",
                       }

code_name_map_sw = {"801010.SI": "农林牧渔（申万）",
                    "801020.SI": "采掘（申万）",
                    "801030.SI": "化工（申万）",
                    "801040.SI": "钢铁（申万）",
                    "801050.SI": "有色金属（申万）",
                    "801080.SI": "电子（申万）",
                    "801110.SI": "家用电器（申万）",
                    "801120.SI": "食品饮料（申万）",
                    "801130.SI": "纺织服装（申万）",
                    "801140.SI": "轻工制造（申万）",
                    "801150.SI": "医药生物（申万）",
                    "801160.SI": "公用事业（申万）",
                    "801170.SI": "交通运输（申万）",
                    "801180.SI": "房地产（申万）",
                    "801200.SI": "商业贸易（申万）",
                    "801210.SI": "休闲服务（申万）",
                    "801230.SI": "综合（申万）",
                    "801710.SI": "建筑材料（申万）",
                    "801720.SI": "建筑装饰（申万）",
                    "801730.SI": "电气设备（申万）",
                    "801740.SI": "国防军工（申万）",
                    "801750.SI": "计算机（申万）",
                    "801760.SI": "传媒（申万）",
                    "801770.SI": "通信（申万）",
                    "801780.SI": "银行（申万）",
                    "801790.SI": "非银金融（申万）",
                    "801880.SI": "汽车（申万）",
                    "801890.SI": "机械设备（申万）"}

index_code_name_map = {'881001.WI': 'WindA',
                       '000300.SH': 'HS300',
                       '000016.SH': 'SZ50',
                       '000905.SH': 'ZZ500',
                      }

NH_index_dict = {
                'A.NH': '南华连大豆指数',
                'AG.NH'	: '南华沪银指数',
                'AL.NH'	: '南华沪铝指数',
                'AP.NH'	: '南华郑苹果指数',
                'AU.NH'	: '南华沪黄金指数',
                'BU.NH'	: '南华沪石油沥青指数',
                'C.NH'	: '南华连玉米指数',
                'CF.NH'	: '南华郑棉花指数',
                'CS.NH'	: '南华连玉米淀粉指数',
                'CU.NH'	: '南华沪铜指数',
                'FG.NH'	: '南华郑玻璃指数',
                'FU.NH'	: '南华沪燃油指数',
                'HC.NH'	: '南华沪热轧卷板指数',
                'I.NH'	: '南华连铁矿石指数',
                'J.NH'	: '南华连焦炭指数',
                'JD.NH'	: '南华连鸡蛋指数',
                'JM.NH'	: '南华连焦煤指数',
                'L.NH'	: '南华连乙烯指数',
                'M.NH'	: '南华连豆粕指数',
                'ME.NH'	: '南华郑甲醇指数',
                'NI.NH'	: '南华沪镍指数',
                'P.NH'	: '南华连棕油指数',
                'PB.NH'	: '南华沪铅指数',
                'PP.NH'	: '南华连聚丙烯指数',
                'RB.NH'	: '南华沪螺钢指数',
                'RM.NH'	: '南华郑菜籽粕指数',
                'RO.NH'	: '南华郑菜油指数',
                'RU.NH'	: '南华沪天胶指数',
                'SC.NH'	: '南华原油指数',
                'SF.NH'	: '南华郑硅铁指数',
                'SM.NH'	: '南华郑锰硅指数',
                'SN.NH'	: '南华沪锡指数',
                'SP.NH'	: '南华纸浆指数',
                'SR.NH'	: '南华郑白糖指数',
                'TA.NH'	: '南华郑精对苯二甲酸指数',
                'TC.NH'	: '南华郑动力煤指数',
                'V.NH'	: '南华连聚氯乙烯指数',
                'Y.NH'	: '南华连豆油指数',
                'ZN.NH'	: '南华沪锌指数',
               }


commodities_delay_num = {
                          # AL
                          '平均价：氧化铝': ('week', 1),
                          '库存期货：铝': ('week', 0),
                          '平均价：铝升贴水：上海物贸': ('week', 0),
                          'LME铝：库存：合计：全球': ('week', 0),
                          'ALD：电解铝：开工率': ('week', 6),
                          'ALD：氧化铝：开工率': ('week', 6),
                          '进口数量：铝土矿：当月值': ('week', 6),
                          '产量：氧化铝：当月值': ('week', 6),
                          '产量：氧化铝：当月同比': ('week', 6),
                          '产量：原铝（电解铝）：当月同比': ('week', 6),
                          '产量：原铝（电解铝）：当月值': ('week', 6),
                          '消费量：精铝：全球：当月值': ('week', 6),
                          # CU
                          '中国铜冶炼厂：粗炼费(TC)': ('week', 1),
                          '中国铜冶炼厂：精炼费(RC)': ('week', 1),
                          'LME铜：库存：合计：全球': ('week', 0),
                          '库存期货：阴极铜': ('week', 0),
                          '平均价：铜升贴水：上海物贸': ('week', 0),
                          '产量：精炼铜（铜）：当月同比': ('week', 6),
                          '进口数量：精炼铜：当月值': ('week', 8),

                           # ZN
                          '库存期货：锌': ('week', 0),
                          'LME锌：库存：合计：全球': ('week', 0),
                          '到厂价：锌精矿': ('week', 0),
                          '锌升贴水': ('week', 0),
                          'ILZSG：全球精炼锌过剩/缺口：当月值': ('week', 8),
                          '产量：锌：当月同比': ('week', 8),

                          # pb
                          '库存期货：铅': ('week', 0),
                          'LME铅：库存：合计：全球': ('week', 0),
                          '长江有色市场：平均价：铅': ('week', 1),
                          '产量：铅：当月同比': ('week', 8),
                          'ILZSG：全球精炼铅过剩/缺口：当月值': ('week', 8),

                          # NI
                          '库存期货：镍': ('week', 0),
                          'LME镍：库存：合计：全球': ('week', 0),
                          '长江有色市场：平均价：镍': ('week', 1),
                          '进口数量：镍矿及精矿：环比': ('week', 8),

                          # FG
                          '浮法玻璃：生产线开工率': ('week', 1),
                          '浮法玻璃：生产线库存': ('week', 1),
                          '浮法玻璃产销余额：产量：当月值': ('week', 6),

                          # P
                          '马来西亚：产量：棕榈油': ('week', 6),
                          '马来西亚：期末库存量：棕榈油': ('week', 6),
                          '马来西亚：出口数量：棕榈油': ('week', 6),
                          '港口库存：棕榈油': ('week', 1),
                          '棕榈油：商业库存：全国': ('week', 1),

                          # Y
                          '进口到港数量：大豆：当期值': ('week', 1),
                          '装船数量：进口大豆：当期值': ('week', 1),
                          '商业库存：豆油': ('week', 1),
                          '注册仓单量：豆油': ('week', 0),
                          '港口库存：进口大豆': ('week', 0),

                          # OI
                          '仓单数量：菜籽油': ('week', 0),
                          '装船数量：进口菜籽油：当期值': ('week', 1),
                          '油菜籽（进口）：CNF到岸价': ('week', 1),
                          '油菜籽（进口）：进口成本价': ('week', 1),

                        }




