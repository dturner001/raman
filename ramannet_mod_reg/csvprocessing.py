import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import train_model_reg as tm


train_in = pd.read_csv("Training_inputs.csv")
#print((train_in.columns[1].split())[0].split("_")[1])

#col name processing function
train_in = train_in.drop(train_in.columns[0], axis=1)

# print(train_in.head())
train_in = train_in.rename(columns = lambda x: x.split()[0].split("_")[1])
#print(train_in.columns)


temps = np.array(train_in.columns)

X = train_in.T.values


X_train, X_Val, Y_train, Y_val = train_test_split(
    X, temps, test_size = 0.2, random_state= 42, shuffle = True
)


print(temps)
print(len(Y_train))

print(f"X_train shape: {X_train.shape}")
print(f"number of temps: {len(temps)}")
path = r"D:\Users\BMT_Admin\Documents\DANIEL\machine_learning_practice\RamanNet\regression model"
mdl, training_history = tm.train_regression_model(X_train, Y_train, X_Val, Y_val, 50, 25, 10, path, plot = True)
