import pandas as pd
import numpy as np


def load_data_pd(fn, tp=''):
    '''Load data file.'''
    print('>> Loading data: {}'.format(fn))
    df = pd.read_csv(fn, delim_whitespace=True, comment='#', header=None)
    df = df.to_numpy()
    return df
