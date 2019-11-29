import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import discriminant_analysis as dis

# read the dataset from a local file
file = "D:\\My Desktop\\WLU\\CP322-Machine Learning\\FP\\dataset\\quality-white.csv"
names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
dataset = pd.read_csv(file, names=names)
# pd.set_option('display.width', None)

# drop useless features
dataset = dataset.drop(['chlorides'], axis=1)
dataset = dataset.drop(['density'], axis=1)

# handle the outliers
for i in range(dataset.index.max()):
    if any([
        dataset.loc[i, 'fixed acidity'] not in range(5, 9),
        dataset.loc[i, 'volatile acidity'] > 0.5,
        dataset.loc[i, 'citric acid'] < 0.2 or dataset.loc[i, 'citric acid'] > 0.7,
        dataset.loc[i, 'free sulfur dioxide'] > 100,
        dataset.loc[i, 'total sulfur dioxide'] > 250,
        dataset.loc[i, 'pH'] < 1.5 or dataset.loc[i, 'pH'] > 3.7,
        dataset.loc[i, 'sulphates'] > 0.8
    ]):
        dataset.drop([i], inplace=True)

# descriptive features & target features
Y = dataset.quality
X = dataset.drop('quality', axis=1)

# training dataset & test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# normalization to [-1, 1]
X_train_scaled = preprocessing.scale(X_train)

# apply the linear discrimination model
model = dis.LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# calculate the accuracy of the prediction
x = np.array(y_pred).tolist()
y = np.array(Y_test).tolist()

accurate = 0
for i in range(len(x)):
    # print(x[i])
    if abs(x[i] - y[i]) <= 1:
        accurate += 1

accuracy = round(float(accurate / len(x)), 4) * 100
print(str(accuracy) + '%')
