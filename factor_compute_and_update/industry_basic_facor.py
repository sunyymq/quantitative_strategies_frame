import copy
import sys
import pandas as pd
import numpy as np
import os
from itertools import chain
from functools import reduce
import statsmodels.api as sm
from datetime import datetime, timedelta
# from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from utility.tool0 import Data, scaler

# 生成根据行业基本面数据得到的行业基本面因子，
# 最后所有的基本面因子都处理成1或-1的数据。


# 价格类因子

def price_type_factor():
    pass



# 行业基本面因子与对应股票的映射关系。
map_dict = {''  }




# 根据基本面因子得分和映射关系，最后由打分模型调用，给相应的股票调整得分，
# 调整得分的时候按照给定的系数来调整。









