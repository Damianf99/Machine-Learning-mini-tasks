import numpy as np
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import random

def perceptron(m, n, display = False):
    w = [0,0,0]
    k = 0
    def f(p, y, w):
        result = sum([(w[i] * p[i]) for i in range(len(p))])
        if result > 0:
            return 1
        else:
            return -1

    def w_fun(w, n, y, p):
        new_p = list(map(lambda p: n*y*p, p))
        new_w = [w[i] + new_p[i] for i in range(len(new_p))]
        return new_w

    def classify(w):
        E = []
        for i in range(len(X_train)):
            clas = f(X_train[i],Y_train[i], w)
            if clas != Y_train[i]:
                E.append(list(X_train[i]) + list(Y_train[i]))
        return E

    X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, n_classes=2, n_samples=m, class_sep = 2)

    X_train = np.hstack((X1, np.ones((X1.shape[0], 1), dtype=X1.dtype)))
    temp_const = X_train[:,:2]
    temp_const = [[1]+list(temp_const[n]) for n in range(len(temp_const))]
    X_train = np.vstack(temp_const)
    Y_train = np.vstack((Y1))
    Y_train[Y_train == 0] = -1

    while True:
        E = classify(w)
        if len(E) != 0:
            it = random.randrange(0,len(E))
            w = w_fun(w,n,E[it][-1],E[it][:3])
        k += 1
        if len(E) == 0:
            break
        if k > 999:
            return w, k
    #Dane w formie wykresu:
    if display == True and m > 1:
        y_line = [((w[0] + w[1] * X1[:, 0][i])/(-w[2])) for i in range(len(X1[:, 0]))]
        plt.scatter(X1[:, 0], X1[:, 1], marker='x', c=Y1, s=25, edgecolor='k')
        plt.plot(X1[:, 0],y_line)
        plt.show()

    return w, k

def launch_perceptron_m(n, start, stop, step, display = False):
    if start >= stop or n < 0.01 or n > 1:
        return None
    w_m_list = []
    k_m_list = []

    for m in range(start, stop+1, step):
        k = 1000
        print(f"Perceptron dla parametrów m = {m} i n = {n}")
        while k == 1000:
            w, k = perceptron(m, n, display) #Ostatni parametr ustawiony na True wyświetla wykresy
        w_m_list.append(w)
        k_m_list.append(k)
        print(f'w = {w}\nk = {k}\n')
    return start, stop, step, k_m_list

def launch_perceptron_n(n, m, step, display = False):
    if n > 0.98 or n < 0.01:
        return None
    w_n_list = []
    k_n_list = []
    start_n = n
    while n < 1:
        k = 1000
        print(f"Perceptron dla parametrów m = {m} i n = {round(n,2)}")
        while k == 1000:
            w, k = perceptron(m, n, display) #Ostatni parametr ustawiony na True wyświetla wykresy
        w_n_list.append(w)
        k_n_list.append(k)
        print(f'w = {w}\nk = {k}\n')
        n += step
    return start_n, k_n_list

def display_plots(start, stop, step, k_m_list, start_n, k_n_list):
    plt.subplot(2,1,1)
    plt.plot(list(range(start,stop+1,step)), k_m_list)
    plt.xlabel("m")
    plt.ylabel("k")
    plt.gca().set_title('Statystki dla liczby przykładów m')

    plt.subplot(2,1,2)
    plt.plot(list(np.linspace(start_n,0.99,len(k_n_list))), k_n_list)
    plt.xlabel("n")
    plt.ylabel("k")
    plt.gca().set_title('Statystki dla współczynnika uczenia n')

    plt.tight_layout(pad=1.0)
    plt.show()

start, stop, step, k_m_list = launch_perceptron_m(0.5, 1, 100, 1, False) #Parametry: n = {0.01 -> 1} (float), początek iteracji dla m (int), koniec iteracji dla m (int), krok (int), Czy wyświetlać plot dla każdego przypadku (bool)
start_n, k_n_list = launch_perceptron_n(0.01, 50, 0.01, False) #Parametry: n = {0.01 -> 0.98} (float), m (int), krok (float) - najlepiej < 0.5, Czy wyświetlać plot dla każdego przypadku (bool)
display_plots(start, stop, step, k_m_list, start_n, k_n_list) #Wyświetla ploty wpływu zmian dla parametrów m i n