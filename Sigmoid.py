import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

file_path = r"C:\features.csv"
data = pd.read_csv(file_path)

feature_vector = ["% I4"]
X = data[feature_vector]
y = data["Label"].values

scaler = StandardScaler()
kf = KFold(n_splits=10, shuffle=True)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=len(feature_vector)))
    α = 0.05
    optimizer = Adam(learning_rate=α)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_split=0.3, epochs=150, verbose = 1)

    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    accuracies.append(test_accuracy)

print(f"Promedio: {np.mean(accuracies):.4f}")
print(f"STD de los rendimientos: {np.std(accuracies):.4f}")

features = ["Mean", "Mode", "Std", "Skew", "Curtosis", "Suma1", "% I4", "% mean-std", "%cur-std", "Mean/std", "Mean/mode", "std/cur", "Mean/curt", "Mode/std", "Mode/curt"]

def scatter():
    i = 1
    for feature in features:
        label_0 = data[data['Label'] == 0][feature]
        label_1 = data[data['Label'] == 1][feature]

        plt.scatter(np.zeros_like(label_0), label_0, alpha=0.7, label='Sin fibras')
        plt.scatter(np.ones_like(label_1), label_1, alpha=0.7, label='Con fibras')
        plt.title(f'Feature{i}')
        plt.ylabel(feature)
        plt.legend()
        plt.show()
        i += 1

scatter()

