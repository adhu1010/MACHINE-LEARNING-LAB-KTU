import numpy as np
def input_mat(x,y):
    arr=np.zeros((x,y))
    print("Enter the matrix :")
    for i in range(x):
        for j in range(y):
            arr[i][j]=int(input())
    return arr
def print_mat(c,x,y):
    print('[')
    for i in range(x):
        print('[',end='')
        for j in range(y):
            print(int(c[i][j]),end='  ')
        print(']')
    print(']')
    
x=int(input ('enter the no of rows 1st matrix :'))
y=int(input ('enter the no of cols 1st matrix :'))

n=int(input ('enter the no of rows 2st matrix :'))
m=int(input ('enter the no of cols 2st matrix :'))


if y==n:
    print("enter matrix 1\n")
    mat1=input_mat(x,y)
    print("enter matrix 2\n")
    mat2=input_mat(n,m)
    c=np.matmul(mat1,mat2)
    print_mat(c,x,m)
    
else:
    print("Matrices cant be multiplied ")