import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

m = 940e6
D = 0.755501
a = 1.44
r0 = 0.131349
hbar = 1973
r = np.linspace(0.001, 2, 1000)
dr = r[1] - r[0]
V = D*(np.exp(-2*a*(r-r0)) - 2*np.exp(-a*(r-r0)))
K = hbar**2/(2*m)
d = 2*K/dr**2 + V[1:-1]
e = -K/dr**2 * np.ones(len(d)-1)
E, u = eigh_tridiagonal(d, e, select='i', select_range=(0,0))
print("Lowest energy =", E[0], "eV")
print("Lowest energy =", E[0]/1e6, "MeV")
plt.plot(r[1:-1], u[:,0])
plt.xlabel("r (Å)")
plt.ylabel("u(r)")
plt.title("Ground State of H₂ Morse Potential")
plt.grid()
plt.show()