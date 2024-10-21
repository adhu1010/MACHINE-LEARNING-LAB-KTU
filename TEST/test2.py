import pandas as pd
def get_matrix():
    n = int(input("enter number of rows :"))
    m = int(input("enter number of columns :"))
    matrix = []
    print (f"enter elements of {n}x{m} matrix :")
    for i in range(n):
        while True:
            try:
                row = list(map(int, input(f"row{i+1}:").split()))
                if len(row)!=m:
                    raise ValueError ("number of elements equal to n ")
                matrix.append(row)
                break
            except ValueError as e:
                print(f"invalid input.enter again")
    return pd.DataFrame(matrix)

def fibonacci():
    fib_sequence = []
    a, b = 0, 1
    for _ in range(100):
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

df=get_matrix()
all_elements = df.values.flatten()
fib=fibonacci()
print("data frame")
print(df)
for i in all_elements:
    if i in fib:
        print(f"{i} is a fibonacci number")
