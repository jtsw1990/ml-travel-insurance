# ml-travel-insurance
fun e2e machine learning project to modeling propensity to claim


## Preprocessing
- removing non positive premiums
- remove ages > 100
- remove duration > 547

## Exploratory data analysis
- 58525 data points after cleaning, 914 claim and ~1.5% claim frequency
- highly imbalanced dataset

- Age seems to have a slight downward effect on claim frequency
- Strange peak of exposure at 36 years
[age_one_way](./assets/freq_age_one_way.png)


- duration has a few unrealistic values upwards of 10 years, to either delete or bin  those
- 
[duration_banded_one_way](./assets/freq_duration_banded_one_way.png)

## Feature engineering
- one hot encoding for categorical
