import numpy as np

def find_pattern(array):
    n = array.shape[0]  # Get the size of the array (n x n)
    max_pattern = None
    max_length = 0
    
    # Iterate over each row
    for i in range(n):
        row = array[i]
        
        # Iterate over all possible lengths of subsequences in the row
        for length in range(2, n + 1):
            # Iterate over all possible starting indices of the subsequence
            for start in range(n - length + 1):
                subsequence = tuple(row[start:start+length])
                count = 0
                
                # Check how many other rows contain this subsequence
                for j in range(n):
                    if i != j and set(subsequence).issubset(set(array[j])):
                        count += 1
                
                # If the current subsequence is found in at least two rows and longer than the current max, update max_pattern
                if count >= 1 and length > max_length:
                    max_pattern = subsequence
                    max_length = length
    
    return max_pattern

# Example usage
n = int(input("Enter the dimension n: "))
array = np.zeros((n, n), dtype=int)

print("Enter the elements row-wise:")
for i in range(n):
    for j in range(n):
        array[i, j] = int(input())

print("\nThe given array is:")
print(array)

# Find the maximum pattern
pattern = find_pattern(array)

if pattern:
    print(f"\nMaximum Pattern Found: {pattern}")
else:
    print("\nNo maximum pattern found.")
