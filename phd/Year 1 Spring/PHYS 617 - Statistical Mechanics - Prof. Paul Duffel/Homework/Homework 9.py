import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

##### CONSTANTS #####
c = 3e8                                         # Speed of light (m/s)
hbar = 6.582e-16                                # Planck's constant (eV/Hz)     |   1.05e-34 (J*s)
k_B = 8.617e-5                                  # Boltzmann constant (eV/K)     |   1.38e-23 (J/K)
# m_e = 9.11e-31                                # Electron mass (GeV/c^2)
# e = 1.6e-19                                   # Elementary charge (C)
N_A = 6.022e23                                  # Avogadro's number (cm^-3)

T = 1e5                                         # Temperature (K)
beta = 1 / (k_B*T)                              # Beta (1/eV)

m = 2829570 / c**2                              # Mass of Helium-4 Atom (eV/c)
zeta_1 = 2.612                                  # Riemann Zeta Function of 3/2
zeta_2 = 1.341                                  # Riemann Zeta Function of 5/2
C = 1/2 * np.pi**-2 * hbar**-3

################## PROBLEM 2 ##################
N = 10000
mu = np.linspace(-10, 0, N)                     # Chemical Potential (eV)
p = np.linspace(0, 1e5 / c, N)                  # Momentum (eV/c)
epsilon = p**2 / (2*m)                          # Energy of Particle (J)

n = np.zeros_like(mu)                           # Number Density
varepsilon = np.zeros_like(mu)                  # Energy density
varepsilon_classical = np.zeros_like(mu)        # Classical energy density


########## PART (a) ##########
for i, u in enumerate(mu):
    # Integrands for n(mu) and varepsilon(mu)
    n_i = 1/(np.exp(beta*(epsilon - u)) - 1)
    varepsilon_i = epsilon/(np.exp(beta*(epsilon - u)) - 1)

    # Integrating over momentum
    n[i] = C * np.trapz(p**2 * n_i, p)
    varepsilon[i] = C * np.trapz(p**2 * varepsilon_i, p)
    # np.quad(lambda x: x**2 * n_i, 0, 1e6)

varepsilon_classical = 3/2 * n * k_B * T

n_crit = (m / (2 * np.pi * hbar**2 * beta))**(3/2) * zeta_1
# n_c = n_0 * ζ1 * (T / T_c)**(3/2) # n_crit at T = T_c
varepsilon_a = 1 - (0.51 / n_crit) * n
T_crit = (N_A * T**(3/2) / n_crit)**(2/3)


# Plot the results
plt.plot(n, varepsilon/varepsilon_classical, label='Energy Density Parametric Function')
plt.plot(n, varepsilon_a, linestyle='--', label='Energy Density Linear Fit')

plt.title('$\\varepsilon(\mu)$ vs. $n(\mu)$')
plt.xlabel('Particle density')
plt.ylabel('Energy density')
plt.grid()
plt.legend()
plt.show()



########## PART (b) ##########
T_n = np.linspace(0, T_crit, N) # Temperature (K)
beta_l = 1 / (k_B*T_n) # 1/eV
n_c = (m / (2 * np.pi * hbar**2 * beta_l))**(3/2) * zeta_1 # Critical density

varepsilon_classical = np.zeros_like(T) # Classical energy density
varepsilon_classical = 3/2 * N_A * k_B * T_n

varepsilon_l = 3/2 * n_c * beta_l**-1 * zeta_2 / zeta_1

# Plot the results
plt.plot(T_n, varepsilon_l/varepsilon_classical, label='Energy Density Parametric Function')

plt.title('$\\varepsilon(T)$ vs. $T$')
plt.xlabel('Temperature')
plt.ylabel('Energy density')
plt.grid()
plt.legend()
plt.show()



########### PART (c) ##########
T = np.linspace(T_crit, 5*T_crit, N) # K
beta = 1 / (k_B*T) # 1/eV

varepsilon_h = 1 - 0.51*(T_crit / T)**(3/2)

# Plot the results
plt.plot(T_n, varepsilon_l/varepsilon_classical, label='Energy Density T < T_crit')
plt.plot(T, varepsilon_h, label='Energy Density T > T_crit')

plt.title('$\\varepsilon(T)$ vs $T$')
plt.xlabel('Temperature')
plt.ylabel('Energy density')
plt.grid()
plt.legend()
plt.show()



########## PART (d) ##########
# Calculate the specific heat
#T = np.linspace(0, 10*T_c, num_points) # K

C_l = np.diff(varepsilon_l)/np.diff(T_n) # heat capacity
C_h = np.diff(varepsilon_h)/np.diff(T) # heat capacity
#C_h = 1.5 * n_a * k * T**(-1.5) * 0.51 * (T_c / T)**(5/2)
#C_h = 1.5 * 0.5 * T**(-5/2) * T_c**(3/2)
#C = np.concatenate((C_l, C_h)) # heat capacity

#T_s = np.linspace(0, 5*T_c, 2*num_points) # K
varepsilon_classical = np.zeros_like(T_n) # Classical energy density
varepsilon_classical = 3/2 * N_A * k_B * T_n
C_classical = np.diff(varepsilon_classical)/np.diff(T_n) # Classical heat capacity


# Plot the results
#plt.plot(T_s[:-1], C/C_classical, label='Heat Capacity Parametric Function')
plt.plot(T_n[:-1], C_l/C_classical, label='Heat Capacity T < T_crit')
plt.plot(T[:-1], C_h + 1.02, label='Heat Capacity T > T_crit')
# Plot the dotted line at the specified height
plt.axhline(y=1, color='r', linestyle='--', label='3/2 * N_A * k_B')
plt.axvline(x=T_crit, color='b', linestyle='--', label = 'T_crit')

plt.title('Heat Capacity vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Heat Capacity')
plt.grid()
plt.legend()
plt.show()