''' module to contain testing scripts '''

from modules.feature_transformer import one_hot_encode
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
