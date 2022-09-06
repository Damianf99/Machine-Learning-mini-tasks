from math import pi
from math import exp
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB

data = np.genfromtxt("wine.data", dtype=str, delimiter=",")

def prepare_data(data):
    X = []
    y = []
    for n in range(len(data)):
        y.append(int(data[n][0]))
    y = np.reshape(y, (len(y), 1))
    data = np.array(data, float)
    for n in range(len(data)):
        X.append(list(data[n, 1:]))
    return X, y

X,y = prepare_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y)

class Bayes(BaseEstimator, ClassifierMixin):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train  # 0, 1, 2
        self.X_test = X_test
        self.y_train = y_train  # 1, 2, 3
        self.y_test = y_test
        self.means = []
        self.std_dev = []

    def fit(self):
        result = []
        classes = np.unique(self.y_train)
        for n in range(len(classes)):
            result.append([])
        for n in range(len(self.X_train)):
            result[int(self.y_train[n])-1].append(self.X_train[n])
        self.means = []
        self.std_dev = []
        for k in range(len(result[0][0])): #13 cech
            temp = []
            for i in range(len(result)): #3 klasy
                suma = 0
                for j in range(len(result[i])): #ilosc elementow w klasie
                    suma += result[i][j][k]
                suma /= len(result[i])
                temp.append(suma)
            self.means.append(temp[:])

        for k in range(len(result[0][0])):
            temp = []
            for i in range(len(result)):
                suma = 0
                for j in range(len(result[i])):
                    suma += ((result[i][j][k] - self.means[k][i])**2)
                suma = sqrt(suma/(len(result[i])-1))
                temp.append(suma)
            self.std_dev.append(temp[:])

    def density(self, x, mean, std_dev):
        e = exp(-((x-mean)**2 / (2 * std_dev**2)))
        return (1 / (sqrt(2 * pi) * std_dev)) * e

    def predict(self):
        classes = np.unique(self.y_train)
        abundance_probab = [list(y_train).count(1)/y_train.shape[0], list(y_train).count(2)/y_train.shape[0], list(y_train).count(3)/y_train.shape[0]]
        result = []
        self.chances = []
        for i in range(len(self.X_test)): #wiersz
            temp = []
            for k in range(len(classes)): #ilość klas
                chance = 1
                for j in range(len(self.X_test[0])): #kolumna
                    chance *= self.density(self.X_test[i][j], self.means[j][k], self.std_dev[j][k])
                chance *= abundance_probab[k]
                temp.append(chance)
            self.chances.append(temp[:])

        for i in range(len(self.chances)):
            max_arg = -9999
            id = 0
            for j in range(len(self.chances[i])):
                if self.chances[i][j] > max_arg:
                    max_arg = self.chances[i][j]
                    id = j+1
            result.append([max_arg,id])
        return result

    def predict_proba(self):
        chances_perc = []
        for i in range(len(self.chances)):
            temp = []
            for j in range(len(self.chances[0])):
                temp.append(self.chances[i][j]/sum(self.chances[i]))
            chances_perc.append(temp[:])
        return chances_perc

def accuracy(predicted, tested):
    counter = 0
    for n in range(len(predicted)):
        if predicted[n][1] == int(tested[n]):
            counter += 1
    return counter/len(predicted) * 100

our_model = Bayes(X_train, X_test, y_train, y_test)
our_model.fit()
pred = our_model.predict()
pred_proba = our_model.predict_proba()
acc = accuracy(pred, our_model.y_test)
print("Predict dla wine:\n")
print(pred,"\n\n")
print("Predict_proba dla wine:\n")
for n in range(len(pred_proba)):
    print(pred_proba[n])

print("\nDokładność wine:",acc,"%\n")

y = np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))
print("Dokładność zaimportowanego GaussianNB dla wine:", ((len(X_test)-((y_test != y_pred).sum()))/(len(X_test)))*100,"%")

def prepare_data_iris(data):
    X = []
    y = []
    d_names = {}
    counter = 1
    for n in range(len(data)):
        if data[n][-1] not in d_names.keys():
            d_names[data[n][-1]] = counter
            counter += 1
    for n in range(len(data)):
        y.append(d_names[data[n][-1]])
    y = np.reshape(y, (len(y), 1))
    data = np.array(data[:, :-1], float)
    for n in range(len(data)):
        X.append(list(data[n, :]))
    return X, y

data = np.genfromtxt("iris.data", dtype=str, delimiter=",")
X,y = prepare_data_iris(data)
X_train, X_test, y_train, y_test = train_test_split(X, y)
our_model = Bayes(X_train, X_test, y_train, y_test)
our_model.fit()
pred = our_model.predict()
pred_proba = our_model.predict_proba()
acc = accuracy(pred, our_model.y_test)
print("\nPredict dla iris:\n")
print(pred,"\n\n")
print("Predict_proba dla iris:\n")
for n in range(len(pred_proba)):
    print(pred_proba[n])

print("\nDokładność iris:",acc,"%\n")

y = np.ravel(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))
print("Dokładność zaimportowanego GaussianNB dla iris:", ((len(X_test)-((y_test != y_pred).sum()))/(len(X_test)))*100,"%")


