''' model inference pipeline uses trained model to infer a result from live data '''

from data_modelling import data_modelling_pipeline
from model_training import model_training_pipeline
from modules.data_reader import read_data
import os
import lightgbm as lgb


# Use this to simulate some live data
df = read_data()

# Read in model, write logic to grab latest
model = lgb.Booster(model_file='models/20220828104916_lgbmc_base.txt')

feature_table, response = data_modelling_pipeline(df, inference=True)

results = model.predict(feature_table)

# This in pratice would pipe into some kind of visualisation software to track results
print(results)
