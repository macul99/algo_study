# depth first search for tree, use it to find subtree sum
# hackerrank test: Cut the Tree
# dfs() show how dfs works
# solve() is the best solution since only travel through the graph once

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
