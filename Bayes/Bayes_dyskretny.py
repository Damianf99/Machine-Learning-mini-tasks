import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

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

bins = 30

model = KBinsDiscretizer(bins, encode='ordinal', strategy='uniform')
model.fit(X)
X = model.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)


class Bayes(BaseEstimator, ClassifierMixin):
    def __init__(self, X_train, X_test, y_train, y_test, laplace=False):
        self.X_train = X_train  # 0, 1, 2
        self.X_test = X_test
        self.y_train = y_train  # 1, 2, 3
        self.y_test = y_test
        self.laplace = laplace
        self.results = []
        self.probab_list = []
        self.predict_list = []

    def fit(self):
        self.results.clear()
        self.probab_list.clear()
        self.predict_list.clear()
        unique_values = np.unique(self.X_train)
        unique_values = np.array(list(map(int, unique_values)))
        self.y_train = self.y_train - 1
        for i in range(len(unique_values)):
            temp = []
            for j in range(np.shape(X_train)[1]):
                temp.append([0, 0, 0])
            self.results.append(temp[:])

        for i in range(np.shape(X_train)[0]):
            for j in range(np.shape(X_train)[1]):
                self.results[int(self.X_train[i, j])][j][int(self.y_train[i])] += 1

        abundance = [list(y_train).count(1), list(y_train).count(2), list(y_train).count(3)]
        for i in range(len(self.results)):  # 3
            upper_temp = []
            for j in range(len(self.results[i])):  # 13
                temp_list = []
                for k in range(len(self.results[i][j])):  # 3
                    if self.laplace == False:
                        temp_list.append(self.results[i][j][k]/abundance[k]) #bez poprawki
                    else:
                        temp_list.append((self.results[i][j][k]+1)/(abundance[k]+len(unique_values))) #z poprawką
                upper_temp.append(temp_list[:])
            self.probab_list.append(upper_temp[:])

    def predict(self):
        abundance_probab = [list(y_train).count(1)/y_train.shape[0], list(y_train).count(2)/y_train.shape[0], list(y_train).count(3)/y_train.shape[0]]
        for i in range(self.X_test.shape[0]):
            chances = []
            for k in range(len(abundance_probab)):
                chance = 1
                for j in range(self.X_test.shape[1]):
                    chance *= self.probab_list[int(self.X_test[i][j])][j][k]
                chance *= abundance_probab[k]
                chances.append(chance)
            self.predict_list.append(chances[:])

        y_ = []
        for i in range(len(self.predict_list)):
            max_val = max(self.predict_list[i])
            y_.append([max_val, self.predict_list[i].index(max_val)+1])
        return y_

    def predict_proba(self):
        probabs = []
        for i in range(len(self.predict_list)):
            temp = []
            for j in range(len(self.predict_list[i])):
                try:
                    temp.append(self.predict_list[i][j]/sum(self.predict_list[i]))
                except:
                    temp.append(0)
            probabs.append(temp[:])
        return probabs

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
print("Predict:\n")
print(pred,"\n\n")
print("Predict_proba:\n")
for n in range(len(pred_proba)):
    print(pred_proba[n])
print("\nDokładność:",acc,"%\n")

our_model = Bayes(X_train, X_test, y_train, y_test, True)
our_model.fit()
pred = our_model.predict()
pred_proba = our_model.predict_proba()
acc = accuracy(pred, our_model.y_test)
print("\nPredict:\n")
print(pred,"\n\n")
print("Predict_proba:\n")
for n in range(len(pred_proba)):
    print(pred_proba[n])
print("\nDokładność:",acc,"%")
