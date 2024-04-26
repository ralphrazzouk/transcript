import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

##### CONSTANTS #####
hbar = 1.05 * 10**(-34)               # Planck's constant (GeV/Hz)
k_B = 1.38 * 10**(-23)              # Boltzmann constant (GeV/K)
m_e = 9.11 * 10**(-31)              # Electron mass (GeV/c^2)
mu_0 = 9 * 10**4                    #
e = 1.6 * 10**(-19)                 # Elementary charge (C)

################## PROBLEM 1 ##################
mu = np.linspace(0, 10*mu_0, 1000)

def n(mu):
    return (2 * m_e * mu)**(3/2) / (3 * np.pi**2 * hbar**3)

def p_0(mu):
    return np.sqrt(2 * m_e * mu)

def P(mu):
    return p_0(mu)**5 / (15 * m_e * np.pi**2 * hbar**3)


plt.plot(np.log10(n(mu * e)), np.log10(P(mu * e)), 'r')
plt.xlabel('$\ln(n(\mu))$')
plt.ylabel('$\ln(P(\mu))$')
plt.show()





################## PROBLEM 2 ##################
T = np.array([10**5, 10**6, 10**7])
beta = 1 / (k_B * T)


def n(mu, beta):
    integrand = []
    for mu_var in mu:
        integrand.append(int.quad(lambda p: p**2 / (1 + np.exp(beta * (p**2/(2 * m_e) - mu_var))), 0, 0.01*k_B)[0] / (np.pi**2 * hbar**3))
    return integrand

def P(mu, beta):
    integrand = []
    for mu_var in mu:
        integrand.append(int.quad(lambda p: p**2 * np.log(1 + np.exp(- beta * (p**2/(2 * m_e) - mu_var))), 0, 0.01*k_B)[0] / (np.pi**2 * hbar**3 * beta))
    return integrand


mu = np.linspace(-mu_0, mu_0, 50000)
plt.plot(np.log10(n(mu*e, beta[0])), np.log10(P(mu*e, beta[0])), 'r', label="$T = 10^5$ K")

mu = np.linspace(-mu_0, mu_0, 50000)
plt.plot(np.log10(n(mu*e, beta[1])), np.log10(P(mu*e, beta[1])), 'g', label="$T = 10^6$ K")

mu = np.linspace(-mu_0, mu_0, 50000)
plt.plot(np.log10(n(mu*e, beta[2])), np.log10(P(mu*e, beta[2])), 'b', label="$T = 10^7$ K")

plt.legend(['$T = 10^5$ K', '$T = 10^6$ K', '$T = 10^7$ K'])
plt.xlabel('$\ln(n(\mu))$')
plt.ylabel('$\ln(P(\mu))$')
plt.show()