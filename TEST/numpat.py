import numpy as np

def find_maximum_pattern(arr):
    max_pattern = None
    max_length = 0
    n = arr.shape[0] 
    
    
    for i in range(n):
        row = arr[i]
        
        
        for length in range(2, len(row) + 1):
            
            for start in range(len(row) + 1 - length):
                subsequence = tuple(row[start:start+length])
                count = 0
                
               
                for j in range(n):
                    if i != j and set(subsequence).issubset(set(arr[j])):
                        count += 1
                
                
                if count >= 1 and length > max_length:
                    max_pattern = subsequence
                    max_length = length
                    nofpat=count
                    print(f"\n Pattern Found: {max_pattern}  repeats  {nofpat}  time")
    
    return (max_pattern,nofpat)


n = int(input("Enter the number of rows : "))
array = np.zeros((n, n), dtype=int)

print("Enter the elements row-wise:")
for i in range(n):
    for j in range(n):
        array[i, j] = int(input())

print("\nThe given matrix is:")
print(array)


max_pattern = find_maximum_pattern(array)

if max_pattern:
    print(f"\nMaximum Pattern Found: {max_pattern[0]}  repeats  {max_pattern[1]} 1 time")
else:
    print("\nNo maximum pattern found.")
