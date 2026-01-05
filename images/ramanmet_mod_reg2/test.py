from train_model import evaluate_regression_model

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras 
import joblib
from keras.models import load_model
from scipy.signal import savgol_filter

def test(data_upper , data_lower):
    model_path = r"regression model\saved_modelreg.keras"

    mdl = load_model(model_path)

    test_in = pd.read_csv("Testing_inputs.csv")
    test_in = test_in[(test_in["Wave Number"]>data_lower) & (test_in["Wave Number"]<data_upper)]
    test_out = pd.read_csv("Testing_outputs.csv")
    test_in = test_in.drop(test_in.columns[0], axis=1)
    X = test_in.T.values
    X = savgol_filter(X, 50, 25,axis = 1, mode = 'interp')
    temps = np.array(test_out.columns,dtype = 'float')
    print(temps)
    temp_scaler = joblib.load(model_path.replace('.keras', '_scaler.joblib'))
    spectra_scaler = joblib.load(model_path.replace('.keras', '_spectra_scaler.joblib'))



    metrics = evaluate_regression_model(mdl, X, temps, temp_scaler,spectra_scaler, 50, 25)
    return metrics

if __name__ == "__main__":
    test(1500, 1100)
