import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

train_in = pd.read_csv("Training_inputs.csv")
train_in = train_in[(train_in["Wave Number"]>1100) & (train_in["Wave Number"]<1500)]



waves = train_in["Wave Number"].values
train_out = pd.read_csv("Training_outputs.csv")

temps = np.array(train_out.columns)
print(temps)

train_in = train_in.drop(train_in.columns[0], axis=1)



#train_in = train_in.rename(columns = lambda x: x.split()[0].split("_")[1])
X = train_in.T.values

spectra_scaler = StandardScaler()
X = spectra_scaler.fit_transform(X)


X = savgol_filter(X,100, 9,axis = 1, mode = 'mirror')

rpets = np.array(train_in.columns)
temps = np.sort(temps.astype(float))

fig,axes = plt.subplots(1,2,figsize=(15, 10))

axes[0].plot(waves, X[0,:])
axes[0].plot(waves, X[1,:])
axes[0].plot(waves, X[2,:])
axes[0].set_title("Filtered Spectra")

axes[1].plot(waves, train_in.iloc[:,0])
axes[1].plot(waves, train_in.iloc[:,1])
axes[1].plot(waves, train_in.iloc[:,2])

plt.show()