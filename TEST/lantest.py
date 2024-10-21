import string
with open('/home/cs-ai-03/adhwaith/TEST/file.txt', 'r') as file:
        text = file.read()
text = text.translate(str.maketrans('', '', string.punctuation))
text = text.lower()
l=text.split()
v=('a','e','i','o','u')
print(l)

for i in l:
    count=0
    for j in i:
        if j in v:
            count+=1
    if count%2==0:
        print(i) 
        print("has",count,end=" ")  
        print("vowels")