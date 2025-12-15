import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_in = pd.read_csv("Training_inputs.csv")

train_out = pd.read_csv("Training_outputs.csv")

temps = np.array(train_out.columns)
print(temps)

train_in = train_in.drop(train_in.columns[0], axis=1)
train_in = train_in.rename(columns = lambda x: x.split()[0].split("_")[1])

rpets = np.array(train_in.columns)
temps = np.sort(temps.astype(float))
print(rpets)
print(temps)
plt.plot(rpets, temps)
plt.show()