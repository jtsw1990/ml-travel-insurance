<div id="top"></div>

# ml-travel-insurance

Fun project to model propensity to claim as a classifcation problem, and serve as a template for a more robust personal ML development framework, and of course to try out new shit.

Data is originally from [Kaggle](https://www.kaggle.com/datasets/mhdzahier/travel-insurance). While the features are simple and the claims response is pretty straightforward, strangely many enthusiasts seem to use sales premium and commission dollars as a feature to predict claim lodgement.

While it is understood that this is a contrived example, it leads to a somewhat unrealistic and circular logic, given the commissions are based on premiums which are in turned based on the risk of a particular profile. So, using premiums to predict claims which are then used again to predict premiums isn't a very reliable strategy in the real world.

## **Table of Contents**
- [ML architecture](ml-architecture)
- [Data quality](#)
- [Data analysis](#)
- [Model results](#)
- [Local developlemt](#)
- [Folder structure](#)


## **Solution architecture**

[!ml_pipeline](/assets/ml_pipeline.svg)

<p align="right">(<a href="#top">back to top</a>)</p>

## **Preprocessing**
- removing non positive premiums
- remove ages > 100
- remove duration > 547

## **Exploratory data analysis**
- 58525 data points after cleaning, 914 claim and ~1.5% claim frequency
- highly imbalanced dataset

- Age seems to have a slight downward effect on claim frequency
- Strange peak of exposure at 36 years
<img src="./assets/freq_age_one_way.png">

- duration has a few unrealistic values upwards of 10 years, to either delete or bin  those
- duration 365 has a high frequency compared to non-annual policies
<img src="./assets/freq_duration_banded_one_way.png">


- destination has high cardinality ~top 20 countries capture ~90% of all data points and claims

## **Feature engineering**
- one hot encoding for categorical
