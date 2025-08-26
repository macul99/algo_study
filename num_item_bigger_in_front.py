# Fenwick Tree versions for different equality handling

def count_bigger_fenwick_strict(nums):
    """
    Fenwick Tree: Count STRICTLY bigger elements (current implementation)
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    i & -i keep only the right most bit of i
    """
    sorted_unique = sorted(set(nums))
    idx_map = {v: i for i, v in enumerate(sorted_unique)}
    n = len(sorted_unique)
    tree = [0] * (n + 2)
    
    def update(i):
        i += 1 # 1-indexed
        while i < len(tree):
            tree[i] += 1
            i += i & -i # moves up the tree hierarchy
    
    def query(i):
        res = 0
        i += 1 # 1-indexed
        while i > 0:
            res += tree[i]
            i -= i & -i # moves down the tree hierarchy
        return res
    
    result = []
    for num in nums:
        idx = idx_map[num]
        # Count elements <= current value, subtract from total to get strictly bigger
        total_seen = len(result)
        elements_le_current = query(idx)
        bigger_count = total_seen - elements_le_current
        result.append(bigger_count)
        update(idx)
    
    return result

def count_bigger_fenwick_or_equal(nums):
    """
    Fenwick Tree: Count bigger OR EQUAL elements
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    i & -i keep only the right most bit of i
    """
    sorted_unique = sorted(set(nums))
    idx_map = {v: i for i, v in enumerate(sorted_unique)}
    n = len(sorted_unique)
    tree = [0] * (n + 2)
    
    def update(i):
        i += 1 # 1-indexed
        while i < len(tree):
            tree[i] += 1
            i += i & -i # moves up the tree hierarchy
    
    def query(i):
        res = 0
        i += 1 # 1-indexed
        while i > 0:
            res += tree[i]
            i -= i & -i # moves down the tree hierarchy
        return res
    
    result = []
    for num in nums:
        idx = idx_map[num]
        # Count elements < current value, subtract from total to get >= elements
        total_seen = len(result)
        if idx > 0:
            elements_less_current = query(idx - 1)
        else:
            elements_less_current = 0
        bigger_or_equal_count = total_seen - elements_less_current
        result.append(bigger_or_equal_count)
        update(idx)
    
    return result

# Test the Fenwick Tree versions
test_nums = [4, 1, 4, 2, 1, 3]
print(f"\\nFenwick Tree Results for {test_nums}:")
print(f"Strictly bigger (Fenwick):    {count_bigger_fenwick_strict(test_nums)}")
print(f"Bigger or equal (Fenwick):    {count_bigger_fenwick_or_equal(test_nums)}")
print(f"Simple strictly bigger:       {count_bigger_strict(test_nums)}")
print(f"Simple bigger or equal:       {count_bigger_or_equal(test_nums)}")

# Verify they match
strict_match = count_bigger_fenwick_strict(test_nums) == count_bigger_strict(test_nums)
equal_match = count_bigger_fenwick_or_equal(test_nums) == count_bigger_or_equal(test_nums)
print(f"\\nStrictly bigger match: {strict_match}")
print(f"Bigger or equal match: {equal_match}")
