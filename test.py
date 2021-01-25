import copy
import sys
import pandas as pd
import numpy as np
import os
from itertools import chain
from functools import reduce
import statsmodels.api as sm
import shelve
from datetime import datetime, timedelta
# from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from utility.tool0 import Data, scaler
from utility.factor_data_preprocess import add_to_panels, align
from utility.relate_to_tushare import generate_months_ends, trade_days
from utility.tool1 import CALFUNC, _calculate_su_simple, parallelcal,  lazyproperty, time_decorator, \
    get_signal_season_value, get_fill_vals, linear_interpolate, get_season_mean_value
from utility.constant import data_dair, root_dair
from utility.tool3 import adjust_months, add_to_panels, append_df


if '__main__' == __name__:

    data = Data()
    style_index = data.style_index










