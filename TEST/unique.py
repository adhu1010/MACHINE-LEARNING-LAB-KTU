import numpy as np
import pandas as pd

n = int(input("Enter the dimension: "))
array = np.zeros((n, n))

print("Enter the values for the array:")
for i in range(n):
    for j in range(n):
        array[i, j] = int(input())

print("\nGenerated Array:\n")
print(array)

df = pd.DataFrame(array, columns=[f'col{i}' for i in range(n)])
print("\nGenerated DataFrame:")
print(df)

def find_unique(df):
    unique_rows = []
    for index, row in df.iterrows():
        is_unique = True
        for other_index, other_row in df.iterrows():
            if index!= other_index and row.equals(other_row):
                is_unique = False
                break
        if is_unique:
            unique_rows.append(row.values)
    unique_df = pd.DataFrame(unique_rows, columns=df.columns)
    return unique_df

non_rep_rows_df = find_unique(df)
print("\nUnique Rows:")
print(non_rep_rows_df)