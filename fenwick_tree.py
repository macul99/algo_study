# Concrete Fenwick Tree Applications with Code Examples

class FenwickTree:
    """Generic Fenwick Tree implementation"""
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        """Add delta to position i (1-indexed)"""
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i
    
    def query(self, i):
        """Get prefix sum from 1 to i"""
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= i & -i
        return res
    
    def range_query(self, left, right):
        """Get sum from left to right (inclusive)"""
        return self.query(right) - self.query(left - 1)

# Application 1: Range Sum Queries
def demo_range_sum():
    print("=== Application 1: Range Sum Queries ===")
    arr = [1, 3, 5, 7, 9, 11]
    n = len(arr)
    ft = FenwickTree(n)
    
    # Build tree
    for i, val in enumerate(arr):
        ft.update(i + 1, val)  # 1-indexed
    
    print(f"Array: {arr}")
    print(f"Sum[1:3]: {ft.range_query(1, 3)}")  # 1+3+5 = 9
    print(f"Sum[2:5]: {ft.range_query(2, 5)}")  # 3+5+7+9 = 24
    
    # Update element
    ft.update(3, 2)  # Add 2 to position 3 (value 5 becomes 7)
    print(f"After adding 2 to position 3:")
    print(f"Sum[1:3]: {ft.range_query(1, 3)}")  # 1+3+7 = 11

demo_range_sum()

# Application 2: Inversion Count
def count_inversions(arr):
    """Count inversions using Fenwick Tree"""
    print("\\n=== Application 2: Inversion Count ===")
    
    # Coordinate compression
    sorted_vals = sorted(set(arr))
    coord_map = {v: i+1 for i, v in enumerate(sorted_vals)}
    
    n = len(sorted_vals)
    ft = FenwickTree(n)
    inversions = 0
    
    print(f"Array: {arr}")
    for i, val in enumerate(arr):
        coord = coord_map[val]
        # Count elements > current value that appeared before
        total_before = i
        elements_le = ft.query(coord)
        inversions += total_before - elements_le
        ft.update(coord, 1)
    
    print(f"Total inversions: {inversions}")
    return inversions

count_inversions([4, 3, 2, 1])  # Should be 6 inversions
count_inversions([1, 3, 2, 4])  # Should be 1 inversion

# Application 3: Dynamic Median Finding
class DynamicMedian:
    """Find median in a stream of numbers"""
    def __init__(self, max_val=1000):
        self.ft = FenwickTree(max_val)
        self.count = 0
        self.max_val = max_val
    
    def add(self, val):
        """Add a number to the stream"""
        self.ft.update(val, 1)
        self.count += 1
    
    def find_median(self):
        """Find current median"""
        if self.count % 2 == 1:
            # Odd count - find middle element
            return self.find_kth((self.count + 1) // 2)
        else:
            # Even count - average of two middle elements
            k1 = self.count // 2
            k2 = k1 + 1
            return (self.find_kth(k1) + self.find_kth(k2)) / 2
    
    def find_kth(self, k):
        """Find k-th smallest element (1-indexed)"""
        left, right = 1, self.max_val
        while left < right:
            mid = (left + right) // 2
            if self.ft.query(mid) >= k:
                right = mid
            else:
                left = mid + 1
        return left

print("\\n=== Application 3: Dynamic Median ===")
dm = DynamicMedian()
numbers = [5, 2, 8, 1, 9, 3]
for num in numbers:
    dm.add(num)
    print(f"Added {num}, median: {dm.find_median()}")

# Application 4: Longest Increasing Subsequence (LIS) Length
def lis_length_fenwick(arr):
    """Find LIS length using Fenwick Tree"""
    print("\\n=== Application 4: LIS Length ===")
    
    # Coordinate compression
    sorted_vals = sorted(set(arr))
    coord_map = {v: i+1 for i, v in enumerate(sorted_vals)}
    
    n = len(sorted_vals)
    ft = FenwickTree(n)
    max_lis = 0
    
    print(f"Array: {arr}")
    for val in arr:
        coord = coord_map[val]
        # Find max LIS ending before current value
        max_before = ft.query(coord - 1) if coord > 1 else 0
        current_lis = max_before + 1
        max_lis = max(max_lis, current_lis)
        
        # Update with current LIS length
        ft.update(coord, current_lis - ft.range_query(coord, coord))
    
    print(f"LIS length: {max_lis}")
    return max_lis

lis_length_fenwick([10, 9, 2, 5, 3, 7, 101, 18])  # Should be 4


# More Advanced Applications

# Application 5: 2D Range Sum (using 2D Fenwick Tree)
class FenwickTree2D:
    """2D Fenwick Tree for rectangle sum queries"""
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, r, c, delta):
        """Add delta to position (r, c)"""
        r += 1  # Convert to 1-indexed
        c += 1
        while r <= self.rows:
            col = c
            while col <= self.cols:
                self.tree[r][col] += delta
                col += col & -col
            r += r & -r
    
    def query(self, r, c):
        """Get sum of rectangle from (0,0) to (r,c)"""
        if r < 0 or c < 0:
            return 0
        r += 1  # Convert to 1-indexed
        c += 1
        res = 0
        while r > 0:
            col = c
            while col > 0:
                res += self.tree[r][col]
                col -= col & -col
            r -= r & -r
        return res
    
    def range_sum(self, r1, c1, r2, c2):
        """Get sum of rectangle from (r1,c1) to (r2,c2)"""
        return (self.query(r2, c2) - 
                self.query(r1-1, c2) - 
                self.query(r2, c1-1) + 
                self.query(r1-1, c1-1))

print("=== Application 5: 2D Range Sum ===")
matrix = [
    [1, 2, 3],
    [4, 5, 6], 
    [7, 8, 9]
]
ft2d = FenwickTree2D(3, 3)

# Build 2D tree
for i in range(3):
    for j in range(3):
        ft2d.update(i, j, matrix[i][j])

print(f"Matrix: {matrix}")
print(f"Sum of rectangle (0,0) to (1,1): {ft2d.range_sum(0, 0, 1, 1)}")  # 1+2+4+5 = 12
print(f"Sum of rectangle (1,1) to (2,2): {ft2d.range_sum(1, 1, 2, 2)}")  # 5+6+8+9 = 28

# Application 6: Count of Smaller Numbers After Self
def count_smaller_after_self(nums):
    """For each element, count how many smaller elements are to its right"""
    print("\\n=== Application 6: Count Smaller After Self ===")
    
    # Process from right to left
    sorted_vals = sorted(set(nums))
    coord_map = {v: i+1 for i, v in enumerate(sorted_vals)}
    
    n = len(sorted_vals)
    ft = FenwickTree(n)
    result = []
    
    print(f"Array: {nums}")
    for i in range(len(nums) - 1, -1, -1):
        val = nums[i]
        coord = coord_map[val]
        
        # Count elements smaller than current value
        smaller_count = ft.query(coord - 1) if coord > 1 else 0
        result.append(smaller_count)
        
        # Add current element to tree
        ft.update(coord, 1)
    
    result.reverse()  # We built it backwards
    print(f"Smaller counts: {result}")
    return result

count_smaller_after_self([5, 2, 6, 1])  # [2, 1, 1, 0]

# Application 7: Range Maximum Query (using coordinate compression)
def range_max_updates(operations):
    """Handle range maximum queries with point updates"""
    print("\\n=== Application 7: Range Max with Updates ===")
    
    # This is a more complex example showing how Fenwick Trees
    # can be adapted for different operations
    
    print("Note: Fenwick Trees naturally support SUM operations.")
    print("For MAX operations, we typically use Segment Trees instead.")
    print("But we can simulate some max operations using coordinate compression.")
    
    # Example: Track maximum value seen so far
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    running_max = []
    current_max = 0
    
    for val in values:
        current_max = max(current_max, val)
        running_max.append(current_max)
    
    print(f"Values: {values}")
    print(f"Running max: {running_max}")

range_max_updates([])

# Application 8: Coordinate Compression for Large Ranges
def demo_coordinate_compression():
    """Show how coordinate compression handles large values"""
    print("\\n=== Application 8: Coordinate Compression ===")
    
    # Large values that would be inefficient to handle directly
    large_nums = [1000000, 5, 999999, 42, 2000000]
    
    # Compress to small range
    sorted_unique = sorted(set(large_nums))
    coord_map = {v: i+1 for i, v in enumerate(sorted_unique)}
    
    print(f"Original values: {large_nums}")
    print(f"Unique sorted: {sorted_unique}")
    print(f"Coordinate mapping: {coord_map}")
    
    # Now we can use a small Fenwick Tree
    ft = FenwickTree(len(sorted_unique))
    
    for val in large_nums:
        coord = coord_map[val]
        ft.update(coord, 1)
        print(f"Added {val} (coord {coord}), prefix sums: {[ft.query(i) for i in range(1, len(sorted_unique)+1)]}")

demo_coordinate_compression()
