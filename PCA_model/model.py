
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Dense, BatchNormalization, concatenate, Lambda, LeakyReLU, Attention
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import functions as fn
from sklearn.preprocessing import StandardScaler

tempscaler = StandardScaler()
pcascaler = StandardScaler()


train_in = pd.read_csv("PCA_model/PCA_training.csv")
train_in = train_in.drop(train_in.columns[3], axis=1)
X =  train_in.values
X = pcascaler.fit_transform(X)

print(X)


train_out = pd.read_csv("Training_outputs.csv")
train_out = np.array(train_out.columns, dtype='float')
print(train_out.shape)
train_out = tempscaler.fit_transform(train_out.reshape(-1,1))


X_train, X_Val, Y_train, Y_val = train_test_split(
    X, train_out, test_size = 0.2, random_state= 31, shuffle = True
)

print(X_train.shape)
print(Y_train.shape)



model = Sequential()

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# checkpoint = ModelCheckpoint(
#     filepath = "PCA_model/saved_modelreg.keras", 
#     verbose=1, 
#     monitor='val_loss',
#     save_best_only=True, 
#     mode='min'
    
# )  



model.fit(X_train, Y_train,
        validation_data=(X_Val, Y_val),
        epochs=50,
        batch_size=32,
        verbose = 1)
        #callbacks=[checkpoint])
model.summary()

test_in = pd.read_csv("PCA_model/PCA_testing.csv")
test_in = test_in.drop(test_in.columns[3], axis=1)
X =  test_in.values
X = pcascaler.transform(X)

test_out = pd.read_csv("Testing_outputs.csv")
test_out = np.array(test_out.columns, dtype='float')
test_out = tempscaler.transform(test_out.reshape(-1,1))

metrics = fn.evaluate_regression_model(model, X, test_out, tempscaler)
print(metrics)