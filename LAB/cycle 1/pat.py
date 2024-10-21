import numpy as np
from collections import defaultdict

def frepeat(arr):
    m,n=arr.shape
    subpattern =[]
    pattern_count = defaultdict(int)
    reapeating_patterns = defaultdict(int)
   
    for i in range (n):
        for j in range (n):
                if arr[i][j] in arr[i]:
                    pattern_count[arr[i][j]] +=1

    def extract(row,length):
        subpatterns = []
        for i in range (len(row)-length+1):
            subpatterns.append(tuple(row[i:i+length]))
        return subpatterns

    for rows in arr:
        for length in range (2,n+2):
            subpatterns= extract(rows,length)
            for subpattern in subpatterns:
               
                pattern_count[subpattern] +=1

    for col in arr.T:
        for length in range (2,m+2):
            subpatterns = extract(col,length)
            for subpattern in subpatterns:
                pattern_count[subpattern] +=1

    repeating_patterns = {k:v for k,v in pattern_count.items() if v>1}
    for pattern, count in repeating_patterns.items():
        print(f"patternv {pattern} appears {count} times ")
    return reapeating_patterns
   
m = int(input('enter the number of rows '))
n = int(input('enter the number of  column'))
matrix = np.empty((n,n),dtype=int)
for i in range (0,m):
   for j in range (0,n):
      matrix[i,j]=int(input())
print(matrix)
patterns = frepeat(matrix)
for pattern, count in patterns.items():
    print(f"pattern {pattern} appears {count}")