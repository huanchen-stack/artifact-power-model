import numpy as np
import pandas as pd
import sys
import os
import pickle
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, SCORERS
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("input.csv")

df.columns = ["nr_ssRsrp", "avg_power", "downlink_mbps", "uplink_mbps"]
X_column = ["nr_ssRsrp", "downlink_mbps", "uplink_mbps"]
Y_column = ["avg_power"]

X = df[X_column].to_numpy()
Y = df[Y_column].to_numpy().reshape(-1)

dtr_tm_model = pickle.load(open("dtr_tm_model.pickle", "rb"))
dtr_vz_mi_model = pickle.load(open("dtr_vz_mi_model.pickle", "rb"))
dtr_vz_mn_model = pickle.load(open("dtr_vz_mn_model.pickle", "rb"))

print(dtr_tm_model.predict(X))
print(dtr_vz_mi_model.predict(X))
print(dtr_vz_mn_model.predict(X))