''' modules to include all preprocessing functions'''

import pandas as pd


def remove_non_positive_premiums(df: pd.DataFrame) -> pd.DataFrame:
    ''' removes negative premiums as they represent biased data points '''

    dataf = df.loc[df['Net Sales'] > 0, :].reset_index(drop=True)

    return dataf


def remove_outlier_ages(df: pd.DataFrame, max_threshold: int) -> pd.DataFrame:
    ''' remove entries with ages > 100 '''
    
    dataf = df.loc[df['Age'] <= max_threshold, :].reset_index(drop=True)

    return dataf


def remove_outlier_duration(df: pd.DataFrame, max_threshold: int, min_threshold: int = 0) -> pd.DataFrame:

    dataf = df.loc[(df['Duration'] <= max_threshold) & (df['Duration'] >= 0), :].reset_index(drop=True)

    return dataf