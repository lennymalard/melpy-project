# %% Data loading
import numpy as np
import pandas as pd

train_dataset = pd.read_csv("data/train.csv", delimiter=",")

X_train = train_dataset.iloc[:750,[2,4,5,9]].values
y_train = train_dataset.iloc[:750,1].values.reshape(-1,1)

X_val = train_dataset.iloc[750:,[2,4,5,9]].values
y_val = train_dataset.iloc[750:,1].values.reshape(-1,1)

# %% Data preprocessing
from melpy.preprocessing import SimpleImputer

si = SimpleImputer()
X_train = si.transform(X_train, 2)
X_train = si.transform(X_train, 3)

X_val = si.transform(X_val, 2)
X_val = si.transform(X_val, 3)

from melpy.preprocessing import StringEncoder

se = StringEncoder()
X_train = se.transform(X_train)
X_val = se.transform(X_val)

X_train = np.delete(X_train, 2, axis=1)
X_val = np.delete(X_val, 2, axis=1)

from melpy.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

# %% Data visualization
import seaborn

df = pd.DataFrame({"Pclass": X_train[:,0].flatten(), "Sex": X_train[:,1].flatten(),
                   "Age": X_train[:,2].flatten(), "Fare": X_train[:,3].flatten(),
                   "Survived": y_train.flatten()})

seaborn.pairplot(df, hue="Survived")

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X_train[:,0], X_train[:,2], c=y_train, cmap="coolwarm")
plt.show()

plt.figure()
axis = plt.axes(projection="3d")
axis.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train, cmap="coolwarm")
plt.show()

# %% Modeling
import melpy.NeuralNetworks as nn

model = nn.Sequential(X_train, y_train, X_val, y_val)

model.layers= [
    nn.Dense(X_train.shape[1], 12),
    nn.ReLU(),
    nn.Dense(12,y_train.shape[1]),
    nn.Sigmoid()
]

model.compile(loss_function=nn.BinaryCrossEntropy(), optimizer=nn.Adam(learning_rate=0.01))

# %% Training
model.fit(epochs=10000, verbose = 1, callbacks = [nn.LiveMetrics(mode=3, f1=0, f2=2)], get_output=True)
model.results()

# %% Save
model.save_params("titanic_parameters")
model.save_histories("titanic_history")



