from app.form_stock_pool_class_分行业遍历 import *


def traversing(para_name, scope, bm=None, s_type=None):
    select_type = None                            # 'total_num' 'twice_sort' None
    bt_or_latest = 'bt'                           # 'bt' 'latest'  'latest_pool_daily'

    method = 'score'
    special_market_dict = None
    score_mode = 'all_same'
    use_risk_model = True

    fd = None
    update_ornot = 'update'  # 'update'   'renew'
    start_date = datetime(2010, 1, 1)
    risk_factor = ['size', 'volatility']

    if not s_type:
        s_type = 'tight'            # 'tight',  #  tight

    if not bm:
        bm = 'ZZ500'

    if bm == 'ZZ500':
        n_max = 300
    elif bm == 'HS300':
        n_max = 150

    para_dict = {
                'is_enhance': True,
                'lamda': 10,
                'turnover': None,  # 0.3,
                'te': None,  # 0.4,
                'industry_expose_control': True,  # 是否行业中性
                'industry_max_expose': 0,  # 0.02,
                'size_neutral': True,
                'in_benchmark': True,
                'in_benchmark_wei': 0,  # 0.8,
                'max_num': n_max,
                's_type': s_type,     #  'tight',  #  tight
                'control_factor': {'size': 0},
                 }
    if para_name not in para_dict:
        print('变量名称错误')
        raise KeyError

    res_df = pd.DataFrame()
    nv_all = pd.DataFrame()
    for val in scope:
        try:
            para_dict[para_name] = val
            if para_name == 'in_benchmark_wei':
                para_dict['in_benchmark'] = False
            res_tmp, nv, last_wei = growth_stock_pool(method=method, score_m=score_mode, select_type=select_type,
                                                      risk_model=use_risk_model, bt_or_latest=bt_or_latest,
                                                      risk_factor=risk_factor, special_market_dict=special_market_dict,
                                                      update_ornot=update_ornot, para_d=para_dict, indus_d=None,
                                                      bm=bm, start_d=start_date, fd_for_scores=fd)
            res_tmp.columns = [val]
            res_df = pd.concat([res_df, res_tmp], axis=1)
            nv_all = pd.concat([nv_all, pd.DataFrame({val: nv})], axis=1)
        except Exception as e:
            continue

    return res_df, nv_all


if '__main__' == __name__:
    # para_name = 'turnover'
    #
    # bm = 'ZZ500'
    # # scope = np.array(range(3, 10, 3)) / 100
    # # for i in scope:
    # #     print(i)
    # scope = [0.35]
    # res_df, nv = traversing(para_name, scope, bm=bm, s_type='loose')
    # save_path = r'D:\Database_Stock\临时'
    # res_df.to_csv(os.path.join(save_path, para_name + '_' + bm + '_评价指标_遍历结果_loose.csv'), encoding='gbk')
    # nv.to_csv(os.path.join(save_path, para_name + '_' + bm + '_净值_遍历结果_loose.csv'), encoding='gbk')

    para_name = 'in_benchmark_wei'
    #
    # bm = 'HS300'
    # scope = np.array(range(80, 100, 5)) / 100
    # for i in scope:
    #     print(i)
    # res_df, nv = traversing(para_name, scope, bm=bm, s_type='loose')
    # save_path = r'D:\Database_Stock\临时'
    # res_df.to_csv(os.path.join(save_path, para_name + '_HS300_评价指标_遍历结果_loose.csv'), encoding='gbk')
    # nv.to_csv(os.path.join(save_path, para_name + '_HS300_净值_遍历结果_loose.csv'), encoding='gbk')
    #
    bm = 'ZZ500'
    scope = np.array(range(80, 100, 10)) / 100
    for i in scope:
        print(i)
    res_df, nv = traversing(para_name, scope, bm=bm, s_type='tight')
    save_path = r'D:\Database_Stock\临时'
    res_df.to_csv(os.path.join(save_path, para_name + '_ZZ500_评价指标_遍历结果_tight.csv'), encoding='gbk')
    nv.to_csv(os.path.join(save_path, para_name + '_ZZ500_净值_遍历结果_tight.csv'), encoding='gbk')

