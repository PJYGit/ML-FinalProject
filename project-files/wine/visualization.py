import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


file = "D:\\My Desktop\\WLU\\CP322-Machine Learning\\FP\\dataset\\modified_dataset.csv"
names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

dataset = pandas.read_csv(file, names=names)
pandas.set_option('display.width', None)

print(dataset.head())
print('...')
print(dataset.tail())

print(dataset.describe())

dataset = dataset.drop(['quality'], axis=1)

dataset.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False)
plt.show()

dataset = dataset.drop(['chlorides'], axis=1)
for i in range(dataset.index.max()):
    if any([
        dataset.loc[i, 'fixed acidity'] not in [5, 6, 7, 8, 9],
        dataset.loc[i, 'volatile acidity'] > 0.5,
        dataset.loc[i, 'citric acid'] < 0.2 or dataset.loc[i, 'citric acid'] > 0.7,
        dataset.loc[i, 'free sulfur dioxide'] > 100,
        dataset.loc[i, 'total sulfur dioxide'] > 250,
        dataset.loc[i, 'pH'] < 1.5 or dataset.loc[i, 'pH'] > 3.7,
        dataset.loc[i, 'sulphates'] > 0.8
    ]):
        dataset.drop([i], inplace=True)

dataset.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False)
plt.show()

dataset = dataset.drop(['density'], axis=1)
scatter_matrix(dataset)
plt.show()

