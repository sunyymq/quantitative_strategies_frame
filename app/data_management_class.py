# 数据管理类，功能包括：
# 1、月度数据的更新，行业数据的生成和处理，
# 2、基础数据的更新：股票财务数据的更新，行情数据的更新，估值数据的更新等等。

import pandas as pd
import os
from datetime import datetime
from utility.tool0 import Data
from utility.relate_to_tushare import update_stock_daily_price, update_adj_factor, update_daily_basic, \
                                      update_future_price, update_stock_future_dat, generate_calendar_month_ends
from juyuan.update_fanance_date import update_all_basic
from utility.download_from_wind import update_macro_data, update_industry_data, update_index_wei, \
    update_f_data_from_wind, update_stock_basic_inform, update_industry_basic_data, update_index_data_daily,\
    update_index_data_monthly
from utility.factor_data_preprocess import compute_future_beta

from utility.constant import data_dair, root_dair


class Data_Management:
    """
    Setting for runnning optimization.
    """

    def __init__(self):
        """"""
        # 回测开始日期
        self.stock_basic_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
        self.industry_date_path = r'D:\pythoncode\IndexEnhancement\行业多因子\中信三级行业'

    # 更新股票基础行情数据
    def update_market_quote_daily(self):

        update_stock_daily_price()  # 日度行情数据
        update_adj_factor()         # 复权数据
        update_daily_basic()        # 日度基本数据
        try:
            update_index_data_daily()
        except Exception as e:
            print('有错误，可能是Wind超限')

        # 暂时用不到这些基础数据，暂不更新
        # # 更新期货的南华指数日线数据
        # update_future_price()
        # # todo 有个bug
        # # 更新股指期货数据
        # update_stock_future_dat()
        # # 计算不同股指合约的beta值
        # compute_future_beta()
        # # 更新行业基本面高频数据，如部分价格、库存、等数据
        # update_industry_basic_data()

    def update_macro_data(self):
        update_macro_data()                      # 宏观数据

    def update_from_wind_monthly(self):

        # 检查是否需要更新
        data = Data()
        index_price_monthly = data.index_price_monthly
        if index_price_monthly.columns[-1].month == datetime.today().month:
            print('数据未更新')
            return 0
        try:
            update_industry_data()
            update_stock_basic_inform()
            update_index_wei()
            update_index_data_monthly()
            update_f_data_from_wind()            # 研发数据
        except Exception as e:
            print('有错误，可能是Wind超限')

    # 更新股票基础财务数据
    def update_financial_data(self):
        data = Data()
        roattm = data.roattm
        latest_date = roattm.columns[-1]
        needed = 1
        if latest_date.month == 3:
            if datetime.today().month <= 7:
                needed = 0

        elif latest_date.month == 6:
            if datetime.today().month <= 10:
                needed = 0

        elif latest_date.month == 9:
            if datetime.today().month <= 4:
                needed = 0

        elif latest_date.month == 12:
            raise ValueError

        if needed == 1:
            try:
                update_all_basic()
            except Exception as e:
                print('暂时链接不上聚源，再试')

        else:
            print('无需更新财务数据')

    # 从月度因子表中删除某个因子
    def del_factor_from_panel(self):
        pass

    # 把一个 截面数据添加到已经有的月度模式存储的文件中
    def add_to_panels(self, dat, f_name, freq_in_dat='M'):
        """说明： 把dat依次插入到panel_path的DF中，插入的列名为f_name, 根据dat的类型是DF还是Series可以判断
        是每次插入的数据不同还是每次插入相同的数据。"""

        print(f'开始添加{f_name}数据到目标文件夹')
        panel = os.listdir(self.stock_basic_path)
        for month_date in panel:
            hased_dat = pd.read_csv(os.path.join(self.stock_basic_path, month_date), engine='python')
            hased_dat = hased_dat.set_index('code')

            # 已有该数据，进入下一个文件
            if f_name in list(hased_dat.columns):
                continue

            # 输入数据为 DataFrame, 那么按列插入
            if isinstance(dat, pd.DataFrame):
                mon_str = month_date.split('.')[0]
                if mon_str in dat.columns:
                    # 当dat中的columns也是str格式，且日期与panel一样时，直接添加
                    if f_name not in hased_dat.columns:
                        hased_dat[f_name] = dat[mon_str]
                else:
                    # 否则，当年、月相同，日不同时，需要变成datetime格式而且还有查找
                    target = datetime.strptime(mon_str, "%Y-%m-%d")
                    # 当dat的columns是datetime格式时
                    if isinstance(dat.columns[0], datetime):
                        if freq_in_dat == 'M':
                            finded = None
                            for col in dat.columns:
                                if col.year == target.year and col.month == target.month:
                                    finded = col
                                    break
                            if finded:
                                if f_name not in hased_dat.columns:
                                    hased_dat[f_name] = dat[finded]
                            else:
                                print('{}该期未找到对应数据'.format(mon_str))
                        if freq_in_dat == 'D':
                            if target in dat.columns:
                                if f_name not in hased_dat.columns:
                                    hased_dat[f_name] = dat[target]
                            else:
                                print('{}该期未找到对应数据'.format(mon_str))
                    else:
                        print('现有格式的还未完善')
                        raise Exception
            # 输入数据为 DataFrame, 那么按列插入
            elif isinstance(dat, pd.Series):
                hased_dat[f_name] = dat[hased_dat.index]

            hased_dat = hased_dat.reset_index('code')
            if 'No' in hased_dat.columns:
                del hased_dat['No']
            hased_dat.index.name = 'No'
            hased_dat.to_csv(os.path.join(self.stock_basic_path, month_date), encoding='gbk')

        print('完毕！')


if __name__ == "__main__":
    data_manage = Data_Management()
    data_manage.update_from_wind_monthly()
    # data_manage.update_market_quote_daily()




