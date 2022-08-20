''' module to contain testing scripts '''

from modules.data_wrangler import remove_non_positive_premiums, remove_outlier_ages, remove_outlier_duration
from modules.feature_transformer import bin_destination, bin_product, one_hot_encode
from modules.product_map import product_mapping
import pytest
import pandas as pd


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


def test_remove_outlier_age(data):

    assert remove_outlier_ages(data, 100)['Age'].max() <= 100
    assert remove_outlier_ages(data, 50)['Age'].max() <= 50
    assert remove_outlier_ages(data, 20)['Age'].max() <= 20


def test_remove_non_positive_premiums(data):

    assert remove_non_positive_premiums(data)['Net Sales'].min() > 0


def test_remove_outlier_duration(data):

    assert remove_outlier_duration(data, 100)['Duration'].max() <= 100
    assert remove_outlier_duration(data, 547)['Duration'].max() <= 547
    assert remove_outlier_duration(data, 20)['Duration'].max() <= 20
    assert remove_outlier_duration(data, 2)['Duration'].max() <= 2


def test_product_mapping(data):

    assert len(bin_product(data, product_mapping).unique()) == 5