# 下面两个因子等权合成一个基金重仓股因子，应该在大消费、制造、TMT板块，周期类板块不适用，且因子处理时不做中性化处理。
@lazyproperty
def Fund_Topten(self):
    mes = generate_months_ends()
    data = Data()
    stock_basic_inform = data.stock_basic_inform

    # 读取基金复权净值数据
    adj_nav = data.fund_adj_nav

    # 读取流通市值数据
    negomv = data.negotiablemv_daily

    # 读取基金重仓股数据
    db = shelve.open(r'D:\Database_Stock\Data\fund\fund_portfolio.db')
    code_dict = db['fund_portfolio_code']
    mkv_dict = db['fund_portfolio_MKV']
    db.close()

    # 基金复权净值与重仓股数据的基金取交集
    mult_funds = set(adj_nav.index) & set(mkv_dict.keys())

    adj_nav = adj_nav.loc[mult_funds, :]

    to_del_keys = [k for k in mkv_dict.keys() if k not in mult_funds]
    if len(to_del_keys) != 0:
        for k in to_del_keys:
            mkv_dict.pop(k)

    # 根据净值数据，选择过去一年净值排名前50%的基金
    new_cols = [col for col in adj_nav.columns if col in mes and col > datetime(2007, 1, 1)]
    adj_nav = adj_nav[new_cols]
    adj_nav = adj_nav.dropna(how='all', axis=0)

    # 得到日期为key, 过去一年的区间收益率排名前50%基金代码list为value的dict
    top_50p_dict = {}
    for i in range(12, len(adj_nav.columns)):
        interval_rat = adj_nav[adj_nav.columns[i]] / adj_nav[adj_nav.columns[i - 12]]
        interval_rat = interval_rat.dropna()
        interval_rat = interval_rat.sort_values(ascending=False)

        top_50p_dict.update({adj_nav.columns[i]: list(interval_rat.index[:int(len(interval_rat) * 0.5)])})

    def date_compare(target, d_list):
        res = None
        for d in d_list:
            if target.year == d.year and target.month == d.month:
                res = d
                break
        return res

    # 根据这些基金的持仓，计算持有股票的市值
    stock_ashare_df = pd.DataFrame(0, index=stock_basic_inform.index, columns=mes)
    for key, hold_df in mkv_dict.items():
        for col in hold_df.columns:

            # 披露时间早于月末时间。
            # 太早的数据删除
            fid = date_compare(col, list(top_50p_dict.keys()))
            if not fid:
                continue

            # 检查该期是否为业绩前50%
            if key not in top_50p_dict[fid]:
                continue

            # 选择出对应的月份
            finded = 0
            for all_col in stock_ashare_df.columns:
                if all_col.year == col.year and all_col.month == col.month:
                    finded = all_col
                    break

            # 太早的数据，如02年的
            if finded == 0:
                continue

            # 一个小bug
            if col.month not in [1, 4, 7, 10]:
                continue

            target_month = stock_ashare_df[finded]

            tmp_se = hold_df[col].dropna()
            # 针对对应的股票，市值相加
            for stock in tmp_se.index:
                if stock in target_month.index:
                    target_month[stock] = target_month[stock] + tmp_se[stock]

            # target_month.sum()
            stock_ashare_df[finded] = target_month

        test = (stock_ashare_df > 0).sum().sum()

    test = (stock_ashare_df > 0).sum()

    # 对于空余的月份，直接复制前值
    for col_n in range(1, len(stock_ashare_df.columns)):
        if stock_ashare_df[stock_ashare_df.columns[col_n]].sum() == 0 and \
                stock_ashare_df[stock_ashare_df.columns[col_n - 1]].sum() != 0:
            stock_ashare_df[stock_ashare_df.columns[col_n]] = stock_ashare_df[stock_ashare_df.columns[col_n - 1]]

    # 比上股票的流通股市值
    stock_ashare_df = stock_ashare_df / 10000

    new_cols = [col for col in negomv.columns if col in mes]
    negomv = negomv[new_cols]
    Topten_to_float_ashare = stock_ashare_df / negomv

    Topten_to_float_ashare = Topten_to_float_ashare.dropna(how='all', axis=1)

    # 得到环比改变数据
    delta_tmp = copy.deepcopy(Topten_to_float_ashare)
    to_del_tmp = [col for col in delta_tmp.columns if col.month not in [1, 4, 7, 10]]

    delta_tmp = delta_tmp.drop(to_del_tmp, axis=1)
    delta_tmp = delta_tmp - delta_tmp.shift(1, axis=1)

    Delta_to_float_ashare = pd.DataFrame(0, index=Topten_to_float_ashare.index, columns=Topten_to_float_ashare.columns)
    Delta_to_float_ashare[delta_tmp.columns] = delta_tmp

    # 对于空余的月份，直接复制前值
    for col_n in range(1, len(Delta_to_float_ashare.columns)):
        if Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n]].sum() == 0 and \
                Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n - 1]].sum() != 0:
            Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n]] = \
                Delta_to_float_ashare[Delta_to_float_ashare.columns[col_n - 1]]

    res_dict = {'Topten_to_float_ashare': Topten_to_float_ashare.fillna(0),
                'Delta_to_float_ashare': Delta_to_float_ashare.fillna(0)
                }

    return res_dict
