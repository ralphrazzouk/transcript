import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

##### CONSTANTS #####
h = 1           # Planck's constant (GeV/Hz)
c = 1           # Speed of light (in natural units)
k_B = 1         # Boltzmann constant (GeV/K)
m_p = 1         # Proton mass (GeV/c^2)
L = 1           # Length of the box (m)





################## PROBLEM 1 ##################
# NON-RELATIVISTIC | RELATIVISTIC | ULTRA-RELATIVISTIC ==> 0.01c | 0.1c - 0.5c | 0.999c
v = np.linspace(0.001*c, 0.999*c, 1000)
lorentz_factor = 1 / np.sqrt(1 - (v/c)**2)
p = lorentz_factor*m_p*v
epsilon = np.sqrt(p**2 + (m_p*c)**2)

# TEMPERATURE => 1 K -> 5 K
T = np.linspace(0, 5, 1000)
beta = 1 / (k_B * T)


##### CONTINUOUS INTEGRAL OF PARTITION FUNCTION
def relativisticPartitionFunction(b):
    integrand = int.quad(lambda p: p**2 * (np.exp(- b * c * np.sqrt(p**2 + (m_p*c)**2))), 0, np.inf)[0]
    return (4*np.pi*L**3)/h**3 * integrand

Z = []
for b in beta:
    z = relativisticPartitionFunction(b)
    Z.append(z)

E = - np.diff(np.log(Z)) / np.diff(beta)


##### DISCRETE SUMMATION OF PARTITION FUNCTION
# def discreteRelativisticPartitionFunction(b):
#     Z = []
#     Z_i = []
#     for j in range(len(b)):
#         for i in range(len(p)):
#             Z_i.append(np.exp(- b[j] * np.sqrt((p[i]**2)*(c**2) + (m_p**2)*(c**4))))
#         Z.append(np.sum(Z_i))
#         Z_i = []
#     return Z

# E_discrete = - np.diff(np.log(discreteRelativisticPartitionFunction(beta)))/np.diff(beta)


plt.plot(T[:-1], E, 'r', label='Relativistic')
plt.plot(T, (3*k_B*T)/2, 'g', label='Non-Relativistic')
plt.plot(T, 3*k_B*T, 'b', label='Ultra-Relativistic')
# plt.plot(T[:-1], E_discrete, 'k', label='Discrete')
plt.legend(['Relativistic', 'Non-Relativistic', 'Ultra-Relativistic', 'Discrete'])
plt.xlabel('Temperature (K)')
plt.ylabel('Energy')
plt.title('Equation of State of Relativistic Monatomic Ideal Gas')
plt.show()


C_V = np.diff(E) / np.diff(T[:-1])

# C_V2 = - beta[:-2]**2 * np.diff(E) / np.diff(beta[:-1])
# plt.plot(T[:-2], C_V2, 'r', label='C_V')

plt.plot(T[:-2], C_V, 'r', label='Relativistic')
plt.plot(T[:-2], 1.5*np.ones(len(C_V)), 'g', label='Non-Relativistic')
plt.plot(T[:-2], 3*np.ones(len(C_V)), 'b', label='Ultra-Relativistic')
plt.legend(['Relativistic', 'Non-Relativistic', 'Ultra-Relativistic', 'Discrete'])
plt.xlabel('Temperature (K)')
plt.ylabel('Molar Heat Capacity')
plt.title('Molar Heat Capacity of Relativistic Monatomic Ideal Gas')
plt.show()





################## PROBLEM 2 ##################
T = np.linspace(0.02, 5, 1000)
beta = 1 / (k_B * T)

m_1 = m_p
m_2 = m_p
mu = (m_1*m_2)/(m_1 + m_2)
r = 0.1
I = mu * r**2

def monatomicPartitionFunction(b):
    coefficient = (L**3)/(h**3) * ((2 * np.pi * m_p) / b)**(3/2)
    summation = 0
    for j in range(0, 1000):
        summation += (2*j + 1) * np.exp(-b * (h/(2*np.pi))**2 * 1/(2*I) * j * (j + 1))

    return coefficient * summation

Z = []
for b in beta:
    z = monatomicPartitionFunction(b)
    Z.append(z)


E = - np.diff(np.log(Z)) / np.diff(beta)

plt.plot(T[:-1], E, 'r', label='Diatomic')
plt.plot(T[:-1], 1.5*k_B*T[:-1], 'g', label='3 dof')
plt.plot(T[:-1], 2.5*k_B*T[:-1], 'b', label='5 dof')
plt.legend(['Diatomic', '3 D.O.F.', '5 D.O.F.'])
plt.xlabel('Temperature $T$ [K]')
plt.ylabel('Energy $E$')
plt.title('Equation of State of Non-relativistic Diatomic Ideal Gas')
plt.show()



C_V = np.diff(E) / np.diff(T[:-1])

plt.plot(T[:-2], C_V, 'r', label='Diatomic')
plt.plot(T[:-2], 1.5*np.ones(len(T[:-2])), 'g', label='3 dof')
plt.plot(T[:-2], 2.5*np.ones(len(T[:-2])), 'b', label='5 dof')
plt.legend(['Diatomic', '3 D.O.F.', '5 D.O.F.'])
plt.xlabel('Temperature $T$ [K]')
plt.ylabel('Molar Heat Capacity $C_V$')
plt.title('Molar Heat Capacity of Non-relativistic Diatomic Ideal Gas')
plt.show()







###### PROBLEM 3 #####
x = np.linspace(0, 10, 1000)
C_V = k_B * x**2 * np.exp(x) / (np.exp(x) + 1)**2

plt.plot(x, C_V, 'r', label='Two-state')
plt.legend(['Two-State'])
plt.xlabel('$\\beta \\Delta E$')
plt.ylabel('Heat Capacity $C_V$')
plt.title('Heat Capacity of a Two-State System')
plt.show()




##### PROBLEM 4 #####
T = [12.833, 16.015, 21.304, 24.100, 31.332, 33.407, 41.319, 50.450, 60.540, 70.070, 80.868, 90.190, 103.085, 116.719, 120.283, 137.805, 151.444, 162.867, 173.316, 180.042, 192.641, 203.157, 210.341, 218.511, 222.107, 232.816, 243.176, 252.541, 263.168, 270.507, 274.134, 277.675, 298.15, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
C_p = [0.000115, 0.000192, 0.000424, 0.000600, 0.00135, 0.00161, 0.00313, 0.00579, 0.01035, 0.01681, 0.02762, 0.04058, 0.06462, 0.1009, 0.1124, 0.1794, 0.2448, 0.3083, 0.3728, 0.4175, 0.5072, 0.5881, 0.6465, 0.7146, 0.7449, 0.8404, 0.9341, 1.0220, 1.1232, 1.1967, 1.2332, 1.2686, 1.462, 1.480, 2.446, 3.242, 3.852, 4.312, 4.660, 4.932, 5.162, 5.380]

a = 1300
N = 2
T_theory = np.linspace(0, 1100, 1000)
C_p_theory = 3 * N * k_B * (a/T_theory)**2 * np.exp(a/T_theory) / (np.exp(a/T_theory) - 1)**2

plt.scatter(T, C_p, c='k')
plt.plot(T_theory, C_p_theory, 'r')
plt.legend(['Data', 'Theory'])
plt.xlabel('Temperature $T$ [K]')
plt.ylabel('Molar Heat Capacity $C_P$ [cal/K]')
plt.title('Molar Heat Capacity of Diamond vs. Theoretical Prediction')
plt.show()