import numpy as np
import random
from matplotlib import pyplot as plt

def N_Queens_Evolution(_n, _pop, _pc, _pm, _gen_max):
    def evaluate(matrix):
            attacks = 0
            for i in range(len(matrix)-1):
                for j in range(i+1,len(matrix)):
                    if(abs(matrix[i]-matrix[j]) == abs(i-j)) or (matrix[i] == matrix[j]):
                        attacks += 1
            return attacks

    def selection(P):
        i = 0
        Pn = []
        while i < pop:
            i1 = random.randint(0, pop-1)
            i2 = random.randint(0, pop-1)
            if i1 != i2:
                if evaluate(np.array(P)[i1,:-1]) <= evaluate(np.array(P)[i2,:-1]):
                    Pn.append(P[i1])
                else:
                    Pn.append(P[i2])
                i += 1
        return Pn

    def get_cross_values(P1, P2, rand_range):
        d1 = {}
        for i in range(len(rand_range)):
            if P1[rand_range[i]] in P1[rand_range[0]:rand_range[-1]+1] and P1[rand_range[i]] in P2[rand_range[0]:rand_range[-1]+1]:
                continue
            if P2[rand_range[i]] not in P1[rand_range[0]:rand_range[-1]+1]:
                d1[P2[rand_range[i]]] = P1[rand_range[i]]
            else:
                id = P1[rand_range[0]:rand_range[-1]+1].index(P2[rand_range[i]])
                j = 0
                while id != -1 and j < len(rand_range):
                    if P2[rand_range[id]] not in P1[rand_range[0]:rand_range[-1]+1]:
                        d1[P2[rand_range[id]]] = P1[rand_range[i]]
                        id = -1
                    else:
                        j += 1
                        id = P1[rand_range[0]:rand_range[-1]+1].index(P2[rand_range[id]])
                        if P2[rand_range[id]] == P1[rand_range[id]]:
                            break
        return d1

    def set_cross_values(P, d, rand_range):
        for i in range(len(P)-1):
            if i in rand_range:
                continue
            if P[i] in d.keys():
                P[i] = d[P[i]]
        return P

    def exchange_mapping(P1, P2, rand_range):
        P1[rand_range[0]:rand_range[-1]+1], P2[rand_range[0]:rand_range[-1]+1] = \
        P2[rand_range[0]:rand_range[-1]+1], P1[rand_range[0]:rand_range[-1]+1]
        return P1, P2

    def cross(P1, P2):
        start_id = random.randint(1, len(P1)-4)
        span_range = random.randint(start_id+1, len(P1)-3)
        rand_range = list(range(start_id, span_range+1))
        d1 = get_cross_values(P1, P2, rand_range)
        d2 = get_cross_values(P2, P1, rand_range)
        P1, P2 = exchange_mapping(P1, P2, rand_range)
        P1 = set_cross_values(P1, d1, rand_range)
        P2 = set_cross_values(P2, d2, rand_range)
        return P1, P2

    def crossover(P):
        i = 0
        while i < pop - 2:
            if random.random() <= pc and P[i] != P[i+1]:
                P[i], P[i+1] = cross(P[i], P[i+1])
            i += 2
        return P

    def mutate(P):
        id = random.sample(list(range(0,len(P)-1)),2)
        P[id[0]], P[id[1]] = P[id[1]], P[id[0]]
        return P

    def mutation(Pn):
        i = 0
        while i < pop:
            if random.random() < pm:
                Pn[i] = mutate(Pn[i])
                i = i + 1
        return Pn

    n = _n
    pop = _pop
    gen = 0
    gen_max = _gen_max
    ff_min = 0
    pc = _pc
    pm = _pm

    P = []
    while len(P) != pop:
        p = random.sample(list(range(1,n+1)), n)
        if p not in P:
            P.append(p)

    for i in range(len(P)):
        P[i].append(evaluate(P[i]))
   
    best = np.argmin(np.array(P)[:,-1])
    best_scores = []
    mean_scores = []

    while gen < gen_max and P[best][-1] > ff_min:
        Pn = crossover(P)
        Pn = mutation(Pn)
        for i in range(len(Pn)):
            Pn[i][-1] = evaluate(np.array(Pn)[i,:-1])
        best = np.argmin(np.array(Pn)[:,-1])
        best_scores.append(Pn[best][-1])
        mean_scores.append(np.mean(np.array(Pn)[:,-1]))
        P = Pn[:]
        gen += 1

    return P[best], gen, best_scores, mean_scores

def draw_plot(gen, best_scores, mean_scores):
    plt.plot(list(range(0,gen)),best_scores)
    plt.plot(list(range(0,gen)),mean_scores)
    plt.legend(['Best Score', 'Mean Score'])
    plt.show()

P, gen, best_scores, mean_scores = N_Queens_Evolution(6, 5, 0.9, 0.5, 1000)
print(P)
print(gen)
draw_plot(gen, best_scores, mean_scores)