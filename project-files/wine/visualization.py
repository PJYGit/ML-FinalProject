import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


file = "D:\\My Desktop\\WLU\\CP322-Machine Learning\\FP\\dataset\\quality-white.csv"
names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

data = pandas.read_csv(file, names=names)
pandas.set_option('display.width', None)

data = data.drop(['quality'], axis=1)


print(data.head())
print('...')
print(data.tail())

print(data.describe())

data.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False)
plt.show()

scatter_matrix(data)
plt.show()

for ele in [5, 6, 7, 9]:
    for i in range(data.index.max()):
        if any([
            data.loc[i, 'fixed acidity'] != int(ele)
        ]):
            data.drop([i], inplace=True)
    data.hist()
    plt.show()

