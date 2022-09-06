import time
from matplotlib import pyplot as plt

def N_queens(n, types):
    if types != 'BFS' and types != 'DFS':
        return 0, 0, 0
    global generated_states, checked_states
    all_arrs = []
    p = [0]*n
    check = False
    generated_states = 0
    checked_states = 0

    def flip_pointers():
        flag = 0
        for k in range(n,0,-1):
            neg = 0
            if p[-k] == n:
                for l in range(len(p)-k,len(p)):
                    if p[l] != n:
                        neg = 1
                if neg == 1:
                    continue
                if flag == 0:
                    if sum(p) == n*n:
                        break
                    p[-k-1] += 1
                p[-k] = 1
                if -k == -1:
                    p[-k] = 0
                flag = 1

    def create_new_elements():
        global generated_states
        res = []
        for i in range(n):
            arr = []
            if p[-1] != n+1:
                p[-1] += 1
            for j in range(n):
                arr.append(p[j])
            generated_states += 1
            if p[-1] == n:
                flip_pointers()
            res.append(arr[:])
        return res

    def check_results(matrix):
        global checked_states
        checked_states += 1
        if (0 in matrix or len(list(set(matrix))) != n):
            if types == "BFS":
                del all_arrs[0]
            else:
                del all_arrs[-1]
            return False
        for i in range(len(matrix)-1):
            for j in range(i+1,len(matrix)):
                if(abs(matrix[i]-matrix[j]) == abs(i-j)) or (matrix[i] == matrix[j]):
                    if types == "BFS":
                        del all_arrs[0]
                    else:
                        del all_arrs[-1]
                    return False
        return True

    while check != True:
        if sum(p) != n*n:
            all_arrs += create_new_elements()
        if types == "BFS":
            check = check_results(all_arrs[0])
        elif types == "DFS":
            check = check_results(all_arrs[-1])

    if types == "BFS":
        return all_arrs[0], generated_states, checked_states
    return all_arrs[-1], generated_states, checked_states

def get_data(k, types):
    gens = []
    checks = []
    times = []
    indices = []

    for n in range(4,k+1):
        time1 = time.time()
        _, generated_states, checked_states = N_queens(n, types)
        time2 = time.time()
        print("Wynik:",_, ", Wygenerowane:",generated_states, ", Sprawdzone:",checked_states, ", Czas:",time2-time1)
        gens.append(generated_states)
        checks.append(checked_states)
        times.append(time2-time1)
        indices.append(n)

    print("Czas łącznie:", sum(times))

    plt.subplot(3,1,1)
    plt.plot(indices, gens)
    plt.xlabel("n")
    plt.ylabel("generated")
    plt.gca().set_title('Liczba stanów wygenerowanych')

    plt.subplot(3,1,2)
    plt.plot(indices, checks)
    plt.xlabel("n")
    plt.ylabel("checked")
    plt.gca().set_title('Liczba stanów sprawdzonych')

    plt.subplot(3,1,3)
    plt.plot(indices, times)
    plt.xlabel("n")
    plt.ylabel("time[s]")
    plt.gca().set_title('Względny czas wykonania')
    plt.tight_layout(pad=1.0)
    plt.show()

#Tutaj podajemy dla ilu elementów mają zostać obliczone wyniki (w zakresie od 4 do n)
#Nie polecam dawać więcej niż 7, ze względu na mocne zapchanie pamięci kolejką (dla n=8 trzeba wygenerować już bardzo dużą ilość elementów)
#DFS działa do 8

print("BFS\n\n")
get_data(7,'BFS') #BFS
print("\n\nDFS\n\n")
get_data(7,'DFS') #DFS

