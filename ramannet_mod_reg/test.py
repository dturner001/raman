from train_model_reg import evaluate_regression_model

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras 
import joblib
from keras.models import load_model

model_path = r"regression model\saved_modelreg.keras"

mdl = load_model(model_path)

test_in = pd.read_csv("Testing_inputs.csv")
test_out = pd.read_csv("Testing_outputs.csv")
test_in = test_in.drop(test_in.columns[0], axis=1)
X = test_in.T.values

temps = np.array(test_out.columns,dtype = 'float')
print(temps)
temp_scaler = joblib.load(model_path.replace('.keras', '_scaler.joblib'))


metrics = evaluate_regression_model(mdl, X, temps, temp_scaler, 50, 25)
print(metrics)