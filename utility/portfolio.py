import pandas as pd
import numpy as np
from math import ceil    # 向上取整
from datetime import datetime
from utility.tool0 import Data


# 回测时的记录股票账户市值变动的类，各种成本设置为类的常量属性，也可通过set_fee函数来设定。

class Portfo:
    def __init__(self, open_p, close_p, initial_asset=100000000, show_detail=False):
        self.open_price = open_p
        self.close_price = close_p
        # 费用和冲击成本
        self.tax = 0.0001
        self.fee = 0.00002
        self.buy_impact_cost = 0.0015
        self.sell_impact_cost = 0.003
        # 初始资产
        self.initial_asset = initial_asset
        self.total_value = initial_asset
        # VNPY项目中是用defaultdict做的持仓，每个value里面再嵌套了defaultdict来记录标的信息，
        # 当数据量大的情况下defaultdict可能比pandas快些
        # 每个截面持有的现金、股票的数量和价格
        self.hold_assets = pd.DataFrame(data=[[initial_asset, 1, 1]], index=['cash'], columns=['num', 'value', 'wei'])
        # 日期
        self.dt = None
        # 历史市值
        self.his_asset = pd.Series()
        # 当期所有股票的价格
        self.price_section = None
        # 开盘价还是收盘价
        self.price_type = None
        # 当期所有股票的权重
        self.wei_section = None
        # 每期的交易费用
        self.fee_se = pd.Series()
        # 每期因卖出冲击成本导致的损失
        self.sell_impact_lost_se = pd.Series()
        # 每期因买入冲击成本导致的损失
        self.buy_impact_lost_se = pd.Series()
        self.show_detail = show_detail

    # 设定各种费率
    def set_fee(self, tax, fee, buy_impact_cost, sell_impact_cost):
        self.tax = tax
        self.fee = fee
        self.buy_impact_cost = buy_impact_cost
        self.sell_impact_cost = sell_impact_cost

    # 下单交易，完成对特定标的买入卖出到组合的固定权重比率的操作
    def order(self, code, target_wei, price=None):
        if not price:
            price = self.price_section[code]
            if pd.isna(price):
                # 若无有效价格，则寻找价格替代
                print('{}在{}无价格数据，价格类型为{}'.format(code, self.dt, self.price_type))
                if self.price_type.upper() == 'OPEN':
                    tmp_p = self.open_price
                elif self.price_type.upper() == 'CLOSE':
                    tmp_p = self.close_price

                loc = np.where(self.dt == tmp_p.columns)[0][0]
                for c in range(loc, len(tmp_p.columns), 1):
                    if not pd.isna(tmp_p.loc[code, tmp_p.columns[c]]):
                        price = tmp_p.loc[code, tmp_p.columns[c]]
                        print('用{}日的价格数据替代'.format(tmp_p.columns[c]))
                        break

                if pd.isna(price):
                    for c in range(loc, 0, -1):
                        if not pd.isna(tmp_p.loc[code, tmp_p.columns[c]]):
                            price = tmp_p.loc[code, tmp_p.columns[c]]
                            print('用{}日的价格数据替代'.format(tmp_p.columns[c]))
                            break

                if pd.isna(price):
                    print('且找不到替代价格，指令无法执行')
                    return -1

                self.price_section[code] = price
                if code in self.hold_assets.index:
                    # total_value的定价使用price_section中的价格，而price_section更新价格时，对nan价格不作替换处理。
                    # 但交易时使用未来价格代替，故价格变动会引起total_value变动
                    # err = self.self_check(fee_isin=True)
                    self.total_value = self.total_value + (price - self.hold_assets.loc[code, 'value']) \
                                       * self.hold_assets.loc[code, 'num']*100
                    self.hold_assets.loc[code, 'value'] = price
                    # err = self.self_check(fee_isin=True)

            # 算上冲击成本之后的买入和卖出价格
            buy_price = price * (1 + self.buy_impact_cost)
            sell_price = price * (1 - self.sell_impact_cost)

        if code not in self.hold_assets.index:
            if target_wei == 0:
                return 0
            if self.show_detail:
                print('买入股票{}'.format(code))
            # 新买入一支股票
            hands = self.total_value * target_wei * (1 - self.fee) / (buy_price * 100)
            hands = int(hands)

            # 确保现金为正
            tmp = self.hold_assets.loc['cash', 'num'] - hands * buy_price * 100 * (1 + self.fee)
            while tmp < 0 or hands <= 0:
                hands = hands - 1
                tmp = self.hold_assets.loc['cash', 'num'] - hands * buy_price * 100 * (1 + self.fee)

            if hands == 0:
                return None

            self.hold_assets.loc['cash', 'num'] = self.hold_assets.loc['cash', 'num'] \
                                                  - hands * buy_price * 100 * (1 + self.fee)

            term = pd.DataFrame(data=[[hands, buy_price, target_wei]], index=[code], columns=['num', 'value', 'wei'])
            self.hold_assets = pd.concat([self.hold_assets, term], axis=0)

            # 交易费用统计-佣金
            fee_tmp = hands * 100 * buy_price * self.fee
            self.add_fee(fee_tmp)
            buy_impact = hands * 100 * price * self.buy_impact_cost
            self.add_buy_impact_loss(buy_impact)

        elif code in self.hold_assets.index:
            if target_wei == 0:
                if self.show_detail:
                    print('清仓{}股票'.format(code))

                cash = self.hold_assets.loc[code, 'num'] * 100 * sell_price * (1 - self.fee - self.tax)
                self.hold_assets.loc['cash', 'num'] = self.hold_assets.loc['cash', 'num'] + cash

                fee_tmp = self.hold_assets.loc[code, 'num'] * 100 * sell_price * (self.fee + self.tax)
                # 在开盘的时候，用开盘价给所有股票定价并计算总市值。但在卖出时，要考虑到冲击成本，则该股票的定价就相应的降低了，
                # 等于冲击城市造成了相对开盘价定价的减值损失。
                impactloss = self.hold_assets.loc[code, 'num'] * 100 * price * self.sell_impact_cost
                self.add_fee(fee_tmp)
                self.add_sell_impact_loss(impactloss)

                # 从hold_assets中删除该股票
                self.hold_assets.drop(code, axis=0, inplace=True)

            elif target_wei > self.wei_section[code]:
                # 权重过小，需要买入部分股票
                if self.show_detail:
                    print('补仓{}股票'.format(code))
                cha = target_wei - self.wei_section[code]
                num_to_buy = cha * self.total_value / (100 * buy_price)
                num_to_buy = int(num_to_buy)

                # 确保现金为正
                tmp = self.hold_assets.loc['cash', 'num'] - num_to_buy * buy_price * 100 * (1 + self.fee)
                while tmp < 0 or num_to_buy == 0:
                    num_to_buy = num_to_buy - 1
                    tmp = self.hold_assets.loc['cash', 'num'] - num_to_buy * buy_price * 100 * (1 + self.fee)

                if num_to_buy == 0:
                    return None

                hased_num = self.hold_assets.loc[code, 'num']
                self.hold_assets.loc[code, 'num'] = hased_num + num_to_buy
                # 新买入的股票因为有冲击成本，等于推高了持股价格。
                self.hold_assets.loc[code, 'value'] = (hased_num*price + num_to_buy*buy_price)/(hased_num + num_to_buy)
                self.hold_assets.loc['cash', 'num'] = self.hold_assets.loc['cash', 'num'] - \
                                                      num_to_buy * buy_price * 100 * (1 + self.fee)

                # 交易费用统计-佣金
                fee_tmp = num_to_buy * 100 * buy_price * self.fee
                self.add_fee(fee_tmp)

                buy_impact = num_to_buy * 100 * price * self.buy_impact_cost
                self.add_buy_impact_loss(buy_impact)

            elif target_wei < self.wei_section[code]:
                # 权重过大，需要卖出部分股票
                if self.show_detail:
                    print('减仓{}股票'.format(code))
                cha = self.wei_section[code] - target_wei
                num_to_sell = ceil(cha * self.total_value / (100 * sell_price))
                num_to_sell = int(num_to_sell)
                self.hold_assets.loc[code, 'num'] = self.hold_assets.loc[code, 'num'] - num_to_sell
                self.hold_assets.loc['cash', 'num'] = self.hold_assets.loc['cash', 'num'] + \
                                                      num_to_sell * sell_price * 100 * (1 - self.fee - self.tax)

                # 卖出股票部分的冲击成本损失
                impactloss = num_to_sell * 100 * price * self.sell_impact_cost
                # 交易费用统计-佣金和印花税
                fee_tmp = num_to_sell * 100 * sell_price * (self.fee + self.tax)
                self.add_fee(fee_tmp)
                self.add_sell_impact_loss(impactloss)

    # 统计交易费用的函数
    def add_fee(self, fee_tmp):
        if self.dt in self.fee_se:
            self.fee_se[self.dt] = self.fee_se[self.dt] + fee_tmp
        else:
            self.fee_se = pd.concat([self.fee_se, pd.Series(0, index=[self.dt])])
            self.fee_se[self.dt] = self.fee_se[self.dt] + fee_tmp

    # 统计卖出冲击成本的函数
    def add_sell_impact_loss(self, impact):
        if self.dt in self.sell_impact_lost_se:
            self.sell_impact_lost_se[self.dt] = self.sell_impact_lost_se[self.dt] + impact
        else:
            self.sell_impact_lost_se = pd.concat([self.sell_impact_lost_se, pd.Series(0, index=[self.dt])])
            self.sell_impact_lost_se[self.dt] = self.sell_impact_lost_se[self.dt] + impact

    # 统计买入冲击成本的函数
    def add_buy_impact_loss(self, impact):
        if self.dt in self.buy_impact_lost_se:
            self.buy_impact_lost_se[self.dt] = self.buy_impact_lost_se[self.dt] + impact
        else:
            self.buy_impact_lost_se = pd.concat([self.buy_impact_lost_se, pd.Series(0, index=[self.dt])])
            self.buy_impact_lost_se[self.dt] = self.buy_impact_lost_se[self.dt] + impact

    # 更新截面的价格和日期
    def update_price_and_dt(self, price_se, dt, price_type):
        self.price_section = price_se
        self.dt = dt
        self.price_type = price_type

    # 根据已经更新的价格信息，重新计算市值、各股票权重。
    def update(self, recoding=False):
        if len(self.hold_assets.index) == 1:
            return 0

        # 先更新hold_assets表中的价格信息
        for i in self.hold_assets.index:
            if i == 'cash':
                continue

            new_p = self.price_section[i]
            if not pd.isna(new_p):
                self.hold_assets.loc[i, 'value'] = new_p

        # 计算总资产市值
        cash_v = self.hold_assets.loc['cash', 'num']
        stock_tmp = self.hold_assets.drop('cash', axis=0)
        stock_v = np.dot((stock_tmp['num'] * 100).values, stock_tmp['value'].values)
        self.total_value = cash_v + stock_v

        if pd.isna(self.total_value):
            print('hh,total_value是nan')

        # 计算各个资产的权重
        for i in self.hold_assets.index:
            if i == 'cash':
                self.hold_assets.loc[i, 'wei'] = self.hold_assets.loc[i, 'num'] / self.total_value
            else:
                num = self.hold_assets.loc[i, 'num']
                price = self.hold_assets.loc[i, 'value']
                self.hold_assets.loc[i, 'wei'] = num * 100 * price / self.total_value
        self.wei_section = self.hold_assets['wei']

        # 检查是否存在权重小于0的错误
        if (self.wei_section < 0).any():
            print('bug')

        # 记录净值曲线数据
        if recoding:
            if self.dt in self.his_asset.index:
                self.his_asset[self.dt] = self.total_value
            else:
                self.his_asset = pd.concat([self.his_asset, pd.Series(self.total_value, index=[self.dt])])

    # 自查数据运算结果
    def self_check(self, fee_isin=True):
        # fee_isin 分两种情况，当在月初调仓是，为了保证计算正确性，需要有交易费用，当在月末重新计算股票权重时，调仓后无交易，
        # 无需加入当月的交易费用。
        if len(self.hold_assets.index) <= 1:
            print('err为0.')
            return 0
        tmp_asset = self.hold_assets

        if self.dt in self.fee_se:
            fee = self.fee_se[self.dt]
        else:
            fee = 0

        if self.dt in self.sell_impact_lost_se:
            impact_sell = self.sell_impact_lost_se[self.dt]
        else:
            impact_sell = 0

        if self.dt in self.buy_impact_lost_se:
            impact_buy = self.buy_impact_lost_se[self.dt]
        else:
            impact_buy = 0

        if not fee_isin:
            fee = 0
            impact = 0

        cash_tmp1 = tmp_asset.loc['cash', :]
        stock_tmp1 = tmp_asset.drop('cash', axis=0)
        stock_asset1 = np.dot((stock_tmp1['num'] * 100).values, stock_tmp1['value'].values)
        err = self.total_value - (stock_asset1 + cash_tmp1['num'] + fee + impact_sell)
        return err

    # 返回股票组合的权重
    def get_stocks_wei(self):
        tmp_wei = self.hold_assets['wei']
        return tmp_wei.drop('cash')



