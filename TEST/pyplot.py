import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris=pd.read_csv('/home/cs-ai-03/adhwaith/Iris2.csv')
iris.plot(kind='scatter',x='SepalLengthCm',y='PetalLengthCm')
plt.xlabel='SepalLength'
plt.ylabel='PetalLength'
plt.grid()