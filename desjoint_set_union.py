# DSU demo
# Handerank Journey-to-moon test
def journeyToMoon(n, astronaut):
    """
    Compute the number of valid pairs of astronauts from different countries.
    Uses Union-Find (Disjoint Set Union) for near-linear performance.

    Args:
        n (int): Number of astronauts (labeled 0..n-1)
        astronaut (Iterable[Tuple[int, int]]): Pairs belonging to the same country

    Returns:
        int: Number of cross-country pairs
    """
    # Union-Find setup
    parent = list(range(n))
    size = [1] * n

    def find(x): # find root node
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Union by size
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra # combine
        size[ra] += size[rb] # combine

    # Union all given pairs (guard against invalid/self pairs)
    for a, b in astronaut:
        if 0 <= a < n and 0 <= b < n and a != b:
            union(a, b)

    # Count component sizes
    comp_sizes = {}
    for i in range(n):
        r = find(i)
        comp_sizes[r] = comp_sizes.get(r, 0) + 1

    sizes = list(comp_sizes.values())
    if len(sizes) <= 1:
        return 0

    # Total pairs C(n,2) minus within-component pairs sum C(sz,2)
    total_pairs = n * (n - 1) // 2
    same_pairs = sum(sz * (sz - 1) // 2 for sz in sizes)
    return total_pairs - same_pairs
