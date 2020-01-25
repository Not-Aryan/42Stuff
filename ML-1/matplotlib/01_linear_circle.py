import matplotlib.pyplot as plt
import matplotlib.axis as ax

import numpy as np
x = np.arange(1,11)
y = 2 * x + 5
plt.title('Matplotlib Demo')
plt.ylabel('y axis caption')
plt.xlabel('x axis caption')
plt.grid(color='blue', linestyle='-', linewidth=.5)


plt.plot(x, y, 'o')
plt.show()