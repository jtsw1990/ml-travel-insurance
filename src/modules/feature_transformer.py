''' module to contain all feature engineering processes '''

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import re


def one_hot_encode(df: pd.DataFrame, categorical_feature: str) -> pd.DataFrame:

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[[categorical_feature]]).toarray()
    encoded_array = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([categorical_feature]))
    encoded_array = encoded_array.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    encoded_array = encoded_array.reset_index(drop=True)
    return encoded_array


def bin_duration(df: pd.DataFrame) -> pd.DataFrame:
    pass