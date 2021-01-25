# 负向指标选股，剔除一些不好的股票
        pool_1 = self.get_stock_pool_by_removing()
        # pool_2 = self.size_filter(top_quantile=0.5)
        # pool_1 = self.pool_inter(pool_1, pool_2)

        # 因子合成
        self.concate_factor()

        # 如果使用打分法，那么需要计算新合成因子的ICIR值，在确定权重部分会使用到
        if self.rov_method == 'score':
            self.compute_icir()

        # 正向选股，打分法或者是回归法，返回股票的打分或者是回归得到的预测收益率，
        stock_pool = self.rov_of_all_stocks(self.rov_method)
        stock_pool_value = copy.deepcopy(stock_pool)

        # 个股过滤，针对一些经过公司研究后的特定股票。
        # stock_pool = self.rps_filter(stock_pool)
        # stock_pool = self.wei_allocation(stock_pool, self.index_wei, 0.1, 50)

        pool_2 = self.select_top_n(stock_pool, top_percent=0.1, by_industry=True)
        pool_intered = self.pool_inter(pool_1, pool_2)
        # financial_pool = self.pool_inter(pool_1, pool_2)
        #
        # price_pool = self.select_stocks_by_price(all_rate=0.2)
        #
        # # pool_intered = price_pool
        # pool_intered = self.pool_inter(financial_pool, price_pool)
        # # 每期的股票数量
        # num = pool_intered.sum()
        #
        # pool_secname = pool_2_secname(pool_intered)
        # pool_secname.to_csv(r'D:\Database_Stock\临时\每期股票名称.csv', encoding='gbk')

        if self.score_mode == 'rps':

            # if self.freq == 'W':
            #     pool_intered_w = append_df(pool_intered, target_feq=self.freq)
            #     # 个股形态过滤，避免选出涨幅过大的股票
            #     pool_intered_w = self.pattern_filter(pool_intered_w)
            #     # 指数过滤，主要是为了放在类似15年的那种大幅的下跌
            #     # pool_intered_w = self.index_filter(pool_intered_w)
            # pool = self.factor_top_n(pool_intered, 'RPS'.upper(), 50, 6)

            # pool = pool_intered
            # tmp_df = pd.DataFrame()
            # for col in pool.columns:
            #     if col in stock_pool_value.columns:
            #         value_tmp = stock_pool_value[col]
            #         tof_tmp = pool[col]
            #         all_tmp = pd.concat([value_tmp, tof_tmp], axis=1)
            #         all_tmp.columns = ['value', 'tof']
            #         all_tmp.loc[all_tmp['tof'] == False, 'value'] = np.nan
            #         tmp_df = pd.concat([tmp_df, pd.DataFrame({col: all_tmp['value']})], axis=1)
            #
            # stock_pool = self.wei_allocation(tmp_df, self.index_wei, 0.1, 50)
            # 等权配置
            stock_pool = self.equal_allocation(pool_intered)
            # self.freq = 'M'
            stock_pool.sum()

            # self.precise_control(tmp_df, stock_pool, self.index_wei)

            # self.freq = 'W'
            # pool_intered.sum()
            # stock_pool = self.rps_select(pool_intered, rps_para=0)