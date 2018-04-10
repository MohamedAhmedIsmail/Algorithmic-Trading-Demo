import pandas as pd

import numpy as np


def get_stock_df(stock_file_name):
    df = pd.read_csv(stock_file_name)
    return df
