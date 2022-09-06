from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import math as m
import numpy as np

y_vals = []

def prepare_data(y):
    y_vals.clear()
    d = {}
    new_y = []
    it = 0
    for n in range(len(y)):
        if y[n] not in d:
            d[y[n]] = it
            it += 1
    y_val = [0] * len(d)
    for n in range(len(d)):
        temp = y_val[:]
        temp[n] = 1
        y_vals.append(temp[:])
    for n in range(len(y)):
        new_y.append(y_vals[d[y[n]]])
    return new_y

def accuracy(y, y_pred):
    score = 0
    for n in range(len(y)):
        score += (y[n] == y_pred[n])
    return score/len(y)

class MLP(object):
    def __init__(self, hidden = 10, epochs = 100, eta = 0.1, shuff = True):
        self.hidden = hidden
        self.epochs = epochs
        self.eta = eta
        self.shuff = shuff
        self.costs = []
        self.accs = []

    def _sigmoid(self, z):
        sig = 1/(1+(m.e**(-z)))
        return sig

    def activate(self, X):
        return X * (1 - X)

    def get_classes(self, y):
        classes = []
        for n in range(y.shape[0]):
            classes.append(y_vals[y[n].argmax()])
        return classes

    def _forward(self, X):
        X = np.array(X)
        sig = X.dot(self.w_h) + self.b_h
        for i in range(len(sig)):
            sig[i] = self._sigmoid(sig[i])

        sig2 = sig.dot(self.w_o) + self.b_o
        for i in range(len(sig2)):
            sig2[i] = self._sigmoid(sig2[i])

        return sig, sig2
        
    def _compute_cost(self, y, out):
        total_loss = 0
        for i in range(len(y)):
            for j in range(len(y[0])):
                total_loss += (y[i][j] * m.log10(out[i,j]) + (1 - y[i][j]) * m.log10(1 - out[i,j]))
        return -total_loss

    def fit(self, X, y, display_outcomes = False):
        X = X.values.tolist()
        self.costs.clear()
        self.accs.clear()
        self.w_h = [[np.random.normal(0,0.1) for i in range(self.hidden)] for j in range(len(X[0]))] # (4, 10)
        self.w_o = [[np.random.normal(0,0.1) for i in range(len(y[0]))] for j in range(self.hidden)] # (10, 3)
        self.b_h = [0 for i in range(self.hidden)] # (10)
        self.b_o = [0 for i in range(len(y[0]))] # (3)
        for i in range(self.epochs):
            if self.shuff == True:
                X, y = shuffle(X, y) #2.1
            a_o_list = []
            for j in range(len(X)):
                a_h, a_o = self._forward(X[j]) #2.2.1
                a_o_list.append(a_o)
                active_a_o = self.activate(a_o) #2.2.2
                delta_o = (a_o - y[j]) * active_a_o #2.2.3
                active_a_h = self.activate(a_h) #2.2.4
                delta_h = (np.dot(delta_o, np.transpose(self.w_o))) * active_a_h #2.2.5
                transposed_X_row = np.reshape(X[j], (len(X[j]),1))
                gradient_w_h = np.dot(transposed_X_row, [delta_h]) #2.2.6
                gradient_b_h = delta_h[:] #2.2.7
                transposed_a_h = np.reshape(a_h, (a_h.shape[0],1))
                gradient_w_o = np.dot(transposed_a_h, [delta_o]) #2.2.8
                gradient_b_o = delta_o[:] #2.2.9
                #2.2.10
                self.w_h -= (gradient_w_h * self.eta)
                self.b_h -= (gradient_b_h * self.eta)
                self.w_o -= (gradient_w_o * self.eta)
                self.b_o -= (gradient_b_o * self.eta)
            #2.3
            a_o_list = np.array(a_o_list)
            cost = self._compute_cost(y, a_o_list)
            self.costs.append(cost)
            y_pred = self.predict(a_o_list, True)
            acc = accuracy(y, y_pred)
            self.accs.append(acc)
            if display_outcomes:
                print(cost)
                print(acc, "\n")

    def predict(self, X, within_class = False):
        if within_class == False:
            _, y_pred = self._forward(X)
        else:
            y_pred = X
        y_pred = self.get_classes(y_pred)
        return y_pred

    def draw_plots(self):
        plt.subplot(2,1,1)
        plt.plot(list(range(self.epochs)), self.costs)
        plt.xlabel("epoki")
        plt.ylabel("f. kosztu")
        plt.gca().set_title('Epoki względem funkcji kosztu')

        plt.subplot(2,1,2)
        plt.plot(list(range(self.epochs)), self.accs)
        plt.xlabel("epoki")
        plt.ylabel("dokładność")
        plt.gca().set_title('Epoki względem dokładności')

        plt.tight_layout(pad=1.0)
        plt.show()

X_iris, y_iris = fetch_openml(name="iris", version=1, return_X_y=True)
y_iris_coded = prepare_data(y_iris)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris_coded, random_state=13)

mlp = MLP()
mlp.fit(X_train, y_train, False) #Żeby wyświetlać koszt i dokładność po każdej iteracji należy ostatni parametr ustwaić na True
y_pred = mlp.predict(X_test)
acc = accuracy(y_test, y_pred)
print("Dokładność dla X_test ze zbioru iris:", acc)
mlp.draw_plots()


X_new_set, y_new_set = fetch_openml(name="Ionosphere", version=1, return_X_y=True)
y_new_set_coded = prepare_data(y_new_set)
X_train, X_test, y_train, y_test = train_test_split(X_new_set, y_new_set_coded, random_state=13)

mlp = MLP()
mlp.fit(X_train, y_train, False) #Żeby wyświetlać koszt i dokładność po każdej iteracji należy ostatni parametr ustwaić na True
y_pred = mlp.predict(X_test)
acc = accuracy(y_test, y_pred)
print("Dokładność dla X_test ze zbioru Ionosphere:", acc)
mlp.draw_plots()