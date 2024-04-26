import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial

N = np.arange(0, 170, 1)
f = ( np.log( np.abs(N+1)) + N**2 * np.log( N / (N+1) ) + N) /2

asymp = np.exp( np.log(factorial(N)) - N*np.log(N) + N - f)
asymp_0 = np.exp(np.log(factorial(N)) - N*np.log(N) + N)
asymp_value = np.ones(170)*1.952

# plt.plot(N, asymp, 'b', N, asymp_value, 'r--')
plt.plot(N, asymp_0, 'g')
plt.show()

