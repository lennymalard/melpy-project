# %% Data loading
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=0)

# %% Data visualization
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, alpha=0.3, cmap="brg")
plt.show()

plt.figure()
axis = plt.axes(projection="3d")
axis.scatter(X_train[:,0],X_train[:,1],X_train[:,2], c=y_train, cmap="brg")
plt.show()

# %% Data preprocessing
from melpy.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

from melpy.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
y_train = ohe.transform(y_train)
y_test = ohe.transform(y_test)

# %% Modeling
import melpy.NeuralNetworks as nn

model = nn.Sequential(X_train, y_train, X_test, y_test)

model.add(nn.Dense(X_train.shape[1], 6, activation=nn.LeakyReLU()))
model.add(nn.Dense(6, y_train.shape[1], activation=nn.Softmax()))

model.compile(cost_function=nn.CategoricalCrossEntropy(), optimizer=nn.SGD(learning_rate=0.001, momentum=0.9))
model.summary()

# %% Training
model.fit(epochs=5000, verbose = 1, callbacks=[nn.LiveMetrics()])
model.results()

# %% Save
model.save_params("iris_parameters")
model.save_histories("iris_metrics")