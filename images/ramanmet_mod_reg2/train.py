import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import train_model as tm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def train_reg(data_upper , data_lower):
    if data_upper == 0:
        data_upper = 3100
        
    
    train_in = pd.read_csv("Training_inputs.csv")
    train_in = train_in[(train_in["Wave Number"]>data_lower) & (train_in["Wave Number"]<data_upper)]
    # wave = train_in["Wave Number"].values
    #dropping wave number column
    train_in = train_in.drop(train_in.columns[0], axis=1)
    #print(train_in.columns)
    #renaming columns to just rpet percentage
    train_in = train_in.rename(columns = lambda x: x.split()[0].split("_")[1])

    train_out = pd.read_csv("Training_outputs.csv")

    temps = np.array(train_out.columns, dtype='float')

    X = train_in.T.values
    X = savgol_filter(X, 200, 40,axis = 1, mode = 'interp')
    # plt.plot(wave, X[0,:])
    # plt.show()
    X_train, X_Val, Y_train, Y_val = train_test_split(
        X, temps,  test_size = 0.2, random_state= 31, shuffle = True
    )




    print(X_Val)

    print(f"X_train shape: {X_train.shape}")
    print(f"number of temps: {len(temps)}")
    path = r"C:\Users\Daniel Turmer\Documents\placement_stuff\raman\regression model\saved_modelreg.keras"
    mdl, training_history = tm.train_regression_model(X_train, Y_train,X_Val, Y_val, 50, 25, 75, path, plot = True)
    #print(mdl.summary())


if __name__ == "__main__":
    train_reg(1500, 1100)