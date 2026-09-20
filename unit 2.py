import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
hbar = 1
m = 1
N = 500
L = 1
x = np.linspace(0, L, N)
dx = x[1] - x[0]
K = hbar**2 / (2*m*dx**2)
V = np.zeros(N-2)
d = 2*K + V
o = -K*np.ones(N-3)
E, psi = eigh_tridiagonal(d, o, select='i',
                          select_range=(0, 0))
print("Particle in box")
print("Numerical E =", E[0])
print("Analytical E =", np.pi**2/2)


x = np.linspace(-2, 2, N)
dx = x[1] - x[0]
K = hbar**2/(2*m*dx**2)
V = np.zeros(N-2)
V[x[1:-1] < -0.5] = 10
V[x[1:-1] > 0.5] = 10
d = 2*K + V
o = -K*np.ones(N-3)
E, psi = eigh_tridiagonal(d, o, select='i',
                          select_range=(0, 0))
print("\nFinite well")
print("Numerical E =", E[0])


x = np.linspace(-5, 5, N)
dx = x[1] - x[0]
K = hbar**2/(2*m*dx**2)
V = 0.5*x[1:-1]**2
d = 2*K + V
o = -K*np.ones(N-3)
E, psi = eigh_tridiagonal(d, o, select='i',
                          select_range=(0, 0))
print("\nHarmonic oscillator")
print("Numerical E =", E[0])
print("Analytical E =", 0.5)
psi = psi[:,0]
plt.plot(x[1:-1], psi)
plt.xlabel("x")
plt.ylabel("Wavefunction")
plt.title("Ground State Wavefunction")
plt.grid()
plt.show()