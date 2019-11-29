import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

file = "D:\\My Desktop\\WLU\\CP322-Machine Learning\\FP\\dataset\\quality-white.csv"
names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

data = pandas.read_csv(file, names=names)
pandas.set_option('display.width', None)

data = data.drop(['chlorides'], axis=1)
data = data.drop(['density'], axis=1)
data = data.drop(['quality'], axis=1)

print(data.describe())

for i in range(data.index.max()):
    if any([
        data.loc[i, 'fixed acidity'] not in range(5, 9),
        data.loc[i, 'volatile acidity'] > 0.5,
        data.loc[i, 'citric acid'] < 0.2 or data.loc[i, 'citric acid'] > 0.7,
        data.loc[i, 'free sulfur dioxide'] > 100,
        data.loc[i, 'total sulfur dioxide'] > 250,
        data.loc[i, 'pH'] < 1.5 or data.loc[i, 'pH'] > 3.7,
        data.loc[i, 'sulphates'] > 0.8
    ]):
        data.drop([i], inplace=True)

print(data.describe())
# print(dataset.groupby('quality').size())
data.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False)
# dataset.hist()
# scatter_matrix(dataset)
plt.show()
'''
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=1)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
'''
