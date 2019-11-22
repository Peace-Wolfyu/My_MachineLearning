# -*- coding:utf-8 -*-  
__author__ = 'Mr.Lin'
__date__ = '2019/11/22 16:29'



import matplotlib.pyplot as plt
import numpy as np
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
t = plt.plot(x, y, marker="x")

'很重要 加上才能显示'
plt.show()
print(t)


