import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import train_model as tm


train_in = pd.read_csv("Training_inputs.csv")
#print((train_in.columns[1].split())[0].split("_")[1])

#col name processing function
train_in = train_in.drop(train_in.columns[0], axis=1)
# print(train_in.head())
train_in = train_in.rename(columns = lambda x: x.split()[0].split("_")[1])
#print(train_in.columns)

X = train_in.T.values
labels = train_in.columns.values
le = LabelEncoder()
Y_encoded = le.fit_transform(labels)
Y_onehot = to_categorical(Y_encoded)

X_train, X_temp, Y_train_onehot, Y_val_temp = train_test_split(
    X, Y_onehot, test_size = 0.3, random_state= 42, stratify= Y_encoded
)

X_val, X_test, Y_val, Y_val_onehot = train_test_split(
    X_temp, Y_val_temp, test_size = 0.3, random_state= 42, stratify= Y_encoded
)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train_onehot shape: {Y_train_onehot.shape}")
print(f"Number of classes: {Y_onehot.shape[1]}")
print(f"Classes: {le.classes_}")
path = r"D:\Users\BMT_Admin\Documents\DANIEL\machine_learning_practice\RamanNet"
mdl, training_history = tm.train_model(X_train, Y_train_onehot, X_val, Y_val_onehot, 50, 25, 500, path)
print(training_history)

#test with x temp and y val temp