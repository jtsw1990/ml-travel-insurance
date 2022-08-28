''' module to contain testing scripts for data processing '''

from modules.data_wrangler import remove_non_positive_premiums, remove_outlier_ages, remove_outlier_duration
import pytest
import pandas as pd


@pytest.fixture
def data():
    return pd.read_csv('data/travel_insurance.csv')


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
