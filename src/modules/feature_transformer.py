''' module to contain all feature engineering processes '''

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import re


def one_hot_encode(df: pd.DataFrame, categorical_feature: str) -> pd.Series:

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[[categorical_feature]]).toarray()
    encoded_array = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out([categorical_feature]))
    encoded_array = encoded_array.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    encoded_array = encoded_array.reset_index(drop=True)
    return encoded_array


def bin_duration(df: pd.DataFrame) -> pd.Series:
    pass


def bin_destination(df: pd.DataFrame, top_threshold: int = 20) -> pd.Series:

    top_countries = df['Destination'].value_counts().head(top_threshold).index.tolist()
    banded_destination = df['Destination'].apply(lambda x: np.where(x in top_countries, x, 'OTHER'))

    return banded_destination


def bin_product(df: pd.DataFrame, product_map: dict) -> pd.Series:

    product_mapped = df['Product Name'].apply(lambda x: product_map[x])

    return product_mapped


def scale_features(df: pd.DataFrame, col_to_scale: str) -> pd.DataFrame:

    scaler = MinMaxScaler()
    scaled_col = pd.DataFrame(scaler.fit_transform(df[[col_to_scale]]), columns=[col_to_scale])

    return scaled_col