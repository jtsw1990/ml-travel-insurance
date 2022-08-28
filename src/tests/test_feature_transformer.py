''' module to contain testing scripts for feature engineering '''

from modules.feature_transformer import bin_destination, bin_product, one_hot_encode, scale_features
from modules.product_map import product_mapping
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def data():
    return pd.read_csv('data/travel_insurance.csv')


def test_one_hot_encode(data):
    ''' test to ensure that the encoded column number equals the unique values of categorical '''

    assert one_hot_encode(data, 'Agency').shape[-1] == len(data['Agency'].unique())
    assert one_hot_encode(data, 'Agency Type').shape[-1] == len(data['Agency Type'].unique())
    assert one_hot_encode(data, 'Product Name').shape[-1] == len(data['Product Name'].unique())
    assert one_hot_encode(data, 'Distribution Channel').shape[-1] == len(data['Distribution Channel'].unique())
    assert one_hot_encode(data, 'Destination').shape[-1] == len(data['Destination'].unique())


def test_bin_destination(data):

    assert bin_destination(data, 20).unique().shape[0] == 21


def test_product_mapping(data):

    assert len(bin_product(data, product_mapping).unique()) == 5


def test_scale_features(data):

    assert np.round(scale_features(data, 'Age')['Age'].max(), 5) == 1
    assert np.round(scale_features(data, 'Age')['Age'].min(), 5) == 0
    assert np.round(scale_features(data, 'Duration')['Duration'].max(), 5) == 1
    assert np.round(scale_features(data, 'Duration')['Duration'].min(), 5) == 0
