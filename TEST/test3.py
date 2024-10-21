import pandas as pd
import string
def fibonacci():
    fib_sequence = []
    a, b = 0, 1
    for _ in range(100):
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

def isprime(n):
    flag=0
    if n==0:
        flag=1
    for j in range(2,n//2):
        if n%j==0:
            flag=1
            break
    if flag==0:
        return True
    else:
        return False
fib=fibonacci()
a={"odd":[],"even":[],"prime":[],"fibbanocci":[],"string":[]}
with open("/home/cs-ai-03/adhwaith/TEST/file.txt","r") as file:
    txt=file.read()
    l=txt.split()
    for i in l:
        if i.isalpha()==True:
            a["string"].append(i)
        elif i.isalnum()==True:
            if int(i)%2==1:
                a["odd"].append(i)
            else:
                a["even"].append(i)
            k=int(i)
            if k in fib:
                a["fibbanocci"].append(i)
            if  isprime(k)==True:
                a["prime"].append(i)    
x=len(a["even"])
y=len(a["fibbanocci"])
z=len(a["odd"])
q=len(a["prime"])
w=len(a["string"])
h=[x,y,z,q,w]
M=h.sort()
Max=h[4]
for i in a.keys():
    for d in range(Max-len(a[i])):
        a[i].append("")
print(Max)
dataframe=pd.DataFrame(a)
print(dataframe)

    