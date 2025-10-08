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

'''
Compare BFS and DFS

Key differences BFS vs DFS:

Strategy
BFS: Level by level (all neighbors first).
DFS: Deep along one branch before backtracking.
Data structure
BFS: Queue (FIFO).
DFS: Stack (implicit via recursion or explicit stack).
Path / optimality
BFS: Finds shortest path (minimum edges) in unweighted graphs.
DFS: Does not guarantee shortest path.
Memory usage (typical)
BFS: Can hold an entire layer (can be large: O(width)).
DFS: Stores just the current path + branching (O(depth)).
When it terminates early
BFS: Good if target is near the start.
DFS: Better if target is likely deep and branching factor is large early.
Use cases
BFS: Shortest paths in unweighted graphs, level order, finding connected components breadth stats.
DFS: Topological sort, cycle detection, detecting articulation points/bridges, backtracking puzzles.
Traversal order example (graph: 1 connected to 2,3; 2 to 4; 3 to 5):
BFS from 1: 1, 2, 3, 4, 5
DFS (one possible): 1, 2, 4, 3, 5
Completeness / termination
BFS: Complete (will find a solution if one exists in finite graph).
DFS: May get stuck in deep/inf loops without visited or depth limit.
Weighted graphs
Neither handles weights; for weighted shortest paths you use Dijkstra (non-negative) or others.
Complexity (both)

Time: O(V + E)
Space: BFS O(V) worst; DFS O(V) worst but often smaller in practice if depth << breadth.
Summary: BFS explores broadly to guarantee shortest unweighted paths; DFS dives deep for structural analysis and backtracking tasks.
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


#### another applcation for solving Hackerank chanllege: Red Knight's Shortest Path
from collections import deque

def printShortestPath():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # visited + store move used to reach this cell
    visited = [[False]*n for _ in range(n)]
    moves = [[None]*n for _ in range(n)]

    xi = int(next(it)); yi = int(next(it))
    xf = int(next(it)); yf = int(next(it))

    # (x, y, distance, move_name_taken_to_get_here)
    q = deque()
    q.append((xi, yi, 0, "empty"))

    cnt = -1

    # Directions: (dx, dy, label)
    dirs = [
        (-2, -1, "UL"),
        (-2,  1, "UR"),
        ( 0,  2, "R"),
        ( 2,  1, "LR"),
        ( 2, -1, "LL"),
        ( 0, -2, "L"),
    ]

    while q:
        x, y, dist, mv = q.popleft()

        if x < 0 or x >= n or y < 0 or y >= n:
            continue
        if visited[x][y]:
            continue

        visited[x][y] = True
        moves[x][y] = mv

        if x == xf and y == yf:
            cnt = dist
            break

        for dx, dy, label in dirs:
            q.append((x + dx, y + dy, dist + 1, label))

    if cnt == -1:
        print("Impossible")
        return

    print(cnt)
    # Reconstruct path
    path = []
    x, y = xf, yf
    while moves[x][y] != "empty":
        mv = moves[x][y]
        path.append(mv)
        if mv == "UL":
            x += 2; y += 1
        elif mv == "UR":
            x += 2; y -= 1
        elif mv == "LR":
            x -= 2; y -= 1
        elif mv == "LL":
            x -= 2; y += 1
        elif mv == "R":
            y -= 2
        elif mv == "L":
            y += 2
    print(" ".join(reversed(path)))
