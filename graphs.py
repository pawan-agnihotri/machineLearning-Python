#!/usr/bin/env python
'''
this is multiline comments
'''
#single line comment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

print ("Hello world")
x = [1,2,3,4,5,6]
y = [8,10,12,14,15,17]
y1 = [18,20,22,24,25,27]
plt.plot(x,y,'g-',label="line")
plt.scatter(x,y,label="skitscat", color="red", marker="x")
#plt.stackplot(x,y,y1, colors=['m','b'])
'''
plt.pie(x,
        labels=["a","b","c","d","e",'f'],
        colors=['m','b','c','b','r','y'],
        startangle=90,
        shadow=True,
        explode=(0,.1,0,0.1,0,0),
        autopct="%1.1f%%")
'''

plt.title("first graph")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
