import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import train_model_reg as tm
from data_processing_reg import filter_spectra

train_in = pd.read_csv("Training_inputs.csv")


#dropping extranious frequencies
# train_in = train_in[(train_in["Wave Number"]>500) & (train_in["Wave Number"]<2000)]

#train_in = train_in.apply(filter_spectra)

#droppign wave number column
train_in = train_in.drop(train_in.columns[0], axis=1)

train_out = pd.read_csv("Training_outputs.csv")




temps = np.array(train_out.columns)

X = train_in.T.values


X_train, X_Val, Y_train, Y_val = train_test_split(
    X, temps, test_size = 0.2, random_state= 31, shuffle = True
)


print(temps)
print(len(Y_train))

print(f"X_train shape: {X_train.shape}")
print(f"number of temps: {len(temps)}")
path = r"C:\Users\Daniel Turmer\Documents\placement_stuff\raman\regression model\saved_modelreg.keras"
mdl, training_history = tm.train_regression_model(X_train, Y_train, X_Val, Y_val, 50, 25, 100, path, plot = True)

