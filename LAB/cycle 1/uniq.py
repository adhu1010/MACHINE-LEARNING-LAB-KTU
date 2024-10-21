import pandas as pd
def get_matrix_from_user():
    n = int(input("enter number of rows"))
    m = int(input("enter number of columns"))
    matrix = []
    print (f"enter elements of {n}x{m} matrix")
    for i in range(n):
        while True:
            try:
                row = list(map(int, input(f"row{i+1}:").split()))
                if len(row)!=m:
                    raise ValueError ("number of elements equal to n")
                matrix.append(row)
                break
            except ValueError as e:
                print(f"invalid input.enter again")
    return pd.DataFrame(matrix)

def find_unique_rows(df):
    all_elements = df.values.flatten()
    element_counts = pd.Series(all_elements).value_counts()
    unique_rows_mask = df.applymap(lambda x:element_counts[x]==1).all(axis=1)
    unique_rows_df = df[unique_rows_mask]
    return unique_rows_df

if __name__ == "__main__":
    df = get_matrix_from_user()
    print("\n original dataframe:")
    print(df)
    unique_rows_df = find_unique_rows(df)
    print("\n  unique rows:")
    print(unique_rows_df)