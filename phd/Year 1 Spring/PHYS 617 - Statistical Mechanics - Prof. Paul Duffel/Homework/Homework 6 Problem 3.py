import numpy as np
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(0.005, 10, N)

z_classical = (np.pi / (4 * x))**(3/2)

n = 1
summand = 0
while (n < N):
    summand += np.exp(-x*n**2)
    n += 1
z_quantum = summand**3


plt.plot(x, z_classical, 'r')
plt.plot(x, z_quantum, 'b')
plt.show()