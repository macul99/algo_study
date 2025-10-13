# depth first search for tree, use it to find subtree sum
# hackerrank test: Cut the Tree
# dfs() show how dfs works
# solve() is the best solution since only travel through the graph once


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

def dfs(root, graph, visited):
    member = [root]
    stack = [root]
    visited[root] = True
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                member.append(neighbor)
                stack.append(neighbor)
    return member

def cutTheTree(data, edges):
    # Write your code here
    from collections import defaultdict
    n = len(data)
    total_sum = sum(data)
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    min_diff = float('inf')
    for u, v in edges:
        if len(graph[u])==1:
            min_diff = min(min_diff, abs(total_sum - 2*data[u-1]))
        elif len(graph[v])==1:
            min_diff = min(min_diff, abs(total_sum - 2*data[v-1]))
        else:
            visited_dic = defaultdict(lambda: False)
            visited_dic[u] = True
            member = dfs(v, graph, visited_dic)
            tmp = sum(data[i-1] for i in member)
            min_diff = min(min_diff, abs(total_sum - 2*tmp))

    return min_diff

def solve():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    # Read node values (1-indexed)
    vals = [0]*(N+1)
    total = 0
    for i in range(1, N+1):
        v = int(next(it)); vals[i] = v; total += v

    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a = int(next(it)); b = int(next(it))
        adj[a].append(b)
        adj[b].append(a)

    # Choose a root with degree > 1 if possible (else 1 for small star/line cases)
    root = 1
    for i in range(1, N+1):
        if len(adj[i]) > 1:
            root = i
            break

    # Iterative post-order: stack entries (node, parent, state)
    # state 0 = first visit, 1 = children processed
    stack = [(root, 0, 0)]
    min_diff = total  # large initial
    subtree_sum = [0]*(N+1)
    while stack:
        node, parent, state = stack.pop()
        if state == 0:
            stack.append((node, parent, 1))
            for nb in adj[node]:
                if nb != parent:
                    stack.append((nb, node, 0))
        else:
            s = vals[node]
            for nb in adj[node]:
                if nb != parent:
                    s += subtree_sum[nb]
            subtree_sum[node] = s
            # Cutting edge above this node would split into s and total - s
            d = total - 2*s
            if d < 0: d = -d
            if d < min_diff:
                min_diff = d

    print(min_diff)


### another example to solve Hankerank chanllenge 'Roads and Libraries'
def roadsAndLibraries(n, c_lib, c_road, cities):
    adj = [[] for _ in range(n + 1)]
    for _, (a, b) in enumerate(cities):
        adj[a].append(b)
        adj[b].append(a)

    visited = [False]*(n + 1)

    # If building a library everywhere is cheaper or equal than a road
    if c_lib <= c_road:
        return c_lib * n

    def dfs(start):
        stack = [start]
        count = 0
        visited[start] = True
        while stack:
            u = stack.pop()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        return count

    cost = 0
    for node in range(1, n + 1):
        if not visited[node]:
            size = dfs(node)
            # One library + (size-1) roads (since c_road < c_lib here)
            cost += c_lib + (size - 1) * c_road
    return cost

'''
Make use BFS and DFS at the same time
Hankerank test: Journey to the Moon
'''
def journeyToMoon(n, astronaut):
    checked = [0]*n
    links = [[] for i in range(n)] # trees
    for a in astronaut: # build up trees
        links[a[0]].append(a[1])
        links[a[1]].append(a[0])

    groups = [] # size for each group

    for i, lk in enumerate(links):
        if checked[i] == 0:
            checked[i] = 1
            if len(lk) == 0: # single node, group of size 1
                groups.append(1)
                continue
            groups.append(1) # new group
            stack = lk[::-1] # BFS - put all linked nodes into stack
            while stack:
                node = stack.pop()
                if checked[node] == 0:
                    checked[node] = 1
                    groups[-1] += 1

                for nxt in links[node]: # DFS - put all linked nodes of child into stack
                    if checked[nxt] == 0:
                        stack.append(nxt)
                        checked[nxt] = 1
                        groups[-1] += 1

    # calculate pairs based on group sizes
    if len(groups) <=1:
        return 0
    
    # if group size is a, b, c, d; total pairs = a*(b+c+d) + b*(c+d) + c*(d)
    csum = [0]
    for i in groups[::-1][:-1]:
        csum.append(csum[-1] + i)

    return sum([a*b for a,b in zip(groups[:-1], csum[::-1][:-1])])
    
