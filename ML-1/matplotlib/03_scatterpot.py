import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

N = 500
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0,0,0)
area = np.pi*3
plt.ylabel('y')
plt.xlabel('x')

plt.title('Scatter plot')

plt.scatter(x,y, area)

plt.show()

