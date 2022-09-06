import time
from matplotlib import pyplot as plt

def N_queens(n, types, h_type):
    if types != 'BFS' and types != 'DFS' and types != 'BestFS' and h_type not in [1,21,22,3]:
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
            if types == "BFS" or types == "BestFS":
                del all_arrs[0]
            elif types == "DFS":
                del all_arrs[-1]
            return False
        for i in range(len(matrix)-1):
            for j in range(i+1,len(matrix)):
                if(abs(matrix[i]-matrix[j]) == abs(i-j)) or (matrix[i] == matrix[j]):
                    if types == "BFS" or types == "BestFS":
                        del all_arrs[0]
                    elif types == "DFS":
                        del all_arrs[-1]
                    return False
        return True

    def h1_function(node,m,i):
        w_rows = []
        for j in range(m):
            if node[j] == 0:
                continue
            elif node[j] <= m/2:
                w_rows.append(m-node[j]+1)
            else:
                w_rows.append(node[j])
        return (m-i) * sum(w_rows)

    def h2_function(node):
        matrix = []
        result = 0
        for i in range(n):
            matrix.append([0]*n)
        for i in range(n):
            if node[i] != 0:
                if h_type == 22:
                    #Wersja szybka:
                    if matrix[node[i]-1][i] != 'X':
                        matrix[node[i]-1][i] = 'H'
                    else:
                        return n*n
                if h_type == 21:
                    #Wersja wolna:
                    matrix[node[i]-1][i] = 'H'
                for j in range(n):
                    for k in range(n):
                        if k == i and matrix[j][k] != 'H':
                            matrix[j][k] = 'X'
                        if j == node[i]-1 and matrix[j][k] != 'H':
                            matrix[j][k] = 'X'
                        if node[i]-1 >= i:
                            if ((i + (node[i]-1)) == (j + k) or (i + (node[i]-1)) == (j-k)) and matrix[j][k] != 'H':
                                matrix[j][k] = 'X'
                        if node[i]-1 < i:
                            if ((i + (node[i]-1)) == (j + k) or (i - (node[i]-1)) == (k-j)) and matrix[j][k] != 'H':
                                matrix[j][k] = 'X'
        for i in range(n):
            result += matrix[i].count('X')
        return result

    def h3_function(node):
        dH = 0
        for i in range(n):
            for j in range(i+1,n):
                if node[i] != node[j]:
                    dH += 1
        return int((n/2) * (n-1) - dH)

    def generate(node, m):
        global generated_states
        id = 0
        result = []
        for i in range(len(node)):
            if node[i] == 0:
                id = i
                break
            if i == len(node)-1:
                return []
        for j in range(1,m+1):
            node[id] = j
            generated_states += 1
            if h_type == 1:
                h = h1_function(node[:],m,id+1)
            if h_type == 21 or h_type == 22:
                h = h2_function(node[:])
            if h_type == 3:
                h = h3_function(node[:])
            result.append((h,node[:]))
        return result

    if types == "BestFS":
        all_arrs.append((0,p))

    while check != True:
        if sum(p) != n*n and (types == "BFS" or types == "DFS"):
            all_arrs += create_new_elements()
        if types == "BFS":
            check = check_results(all_arrs[0])
        elif types == "DFS":
            check = check_results(all_arrs[-1])
        elif types == "BestFS":
            if h_type == 1:
                all_arrs += generate(all_arrs[0][1], n)
                all_arrs.sort()
                check = check_results(all_arrs[0][1])
            else:
                all_arrs.sort()
                all_arrs += generate(all_arrs[0][1], n)
                check = check_results(all_arrs[0][1])

    if types == "BFS" or types == "BestFS":
        return all_arrs[0], generated_states, checked_states
    return all_arrs[-1], generated_states, checked_states

def get_data(k, types, h_type = 1):
    gens = []
    checks = []
    times = []
    indices = []

    for n in range(4,k+1):
        time1 = time.time()
        _, generated_states, checked_states = N_queens(n, types, h_type)
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

#Argumenty podajemy w następujący sposób:
#get_data(n, 'nazwa_algorytmu', h)
#Gdzie:
#n - liczba elementów do wygenerowanie (od 4 do n)
#'nazwa_algorytmu' - 'BFS'/'DFS'/'BestFS'
#h (opcjonalnie -> jeżeli nie podamy to zawsze = 1) - 1/21/22/3 odpowiednio dla h1/h2/h3
#Dla h2 istnieją dwie wersje - 21 (wolniejsza) i 22 (szybsza)

print("BFS\n\n")
get_data(6,'BFS') #BFS

print("\n\nDFS\n\n")
get_data(6,'DFS') #DFS

print("\n\nBestFS h1\n\n")
get_data(6,'BestFS') #BestFS h1

print("\n\nBestFS h2\n\n")
get_data(6,'BestFS',22) #BestFS h2

print("\n\nBestFS h3\n\n")
get_data(6,'BestFS',3) #BestFS h3

