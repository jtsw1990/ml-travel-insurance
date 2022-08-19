import pandas as pd


def read_data() -> pd.DataFrame:
    return pd.read_csv('data/travel_insurance.csv')
