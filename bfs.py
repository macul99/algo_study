# Breadth-First Search tree search
'''
commonly used for:

Finding the shortest path in unweighted graphs
Exploring all nodes in a connected component
Solving puzzles and mazes

BFS(graph, start):  
    create a queue Q  
    mark start as visited and enqueue it into Q  
    while Q is not empty:  
        node = Q.dequeue()  
        for each neighbor of node:  
            if neighbor is not visited:  
                mark neighbor as visited  
                enqueue neighbor into Q  
'''

# example to solve "Gena Playing Hanoi" chanllenge from hackerank
N = 10
a = [0] * (2 << (2 * N + 1))
b = [0] * 4
n = int(input())
K = 0
t = 0
'''
A length-n sequence over {1,2,3,4} is packed into a single integer using 2 bits per position (little‑endian by position t: (x-1) << (2*t)).
Array a[] stores distance (steps) from start; a[state] = 0 means unvisited (start is set to 1 so distance = a[state]-1 when reached).
'''
for x in map(int, input().split()):
    K |= ((x - 1) << (t * 2))
    t += 1
a[0] = 1
Q = [0] # BFS queue
ind = 0
while True:
    x = Q[ind]
    ind += 1
    if x == K:
        print(a[x] - 1)
        break
    for i in range(4): # 4 rods
        b[i] = 1000 # assume empty rod first (sentinel)
    for i in range(n - 1, -1, -1):
        b[(3 & (x >> (i * 2)))] = i # the topmost (smallest) disk in each rod
    for i in range(4): # try all possible moves
        for j in range(4):
            if b[i] < b[j]: # move smaller disk onto larger disk
                y = x + ((j - i) << (b[i] * 2))
                if a[y] == 0:
                    a[y] = a[x] + 1
                    Q.append(y)
