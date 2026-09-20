import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
m = 940
h = 197.3
k = 100
r = np.linspace(0.001, 5, 1000)
dr = r[1] - r[0]
K = h**2 / (2*m)
for b in [0, 10, 30]:
    V = 0.5*k*r**2 + b*r**3/3
    d = 2*K/dr**2 + V
    o = -K/dr**2
    E, u = eigh_tridiagonal(d, np.full(len(r)-1, o),
                            select='i', select_range=(0, 0))
    print("b =", b, " E =", round(E[0], 3), "MeV")
    u = u[:,0]
    u = u / np.sqrt(np.sum(u**2)*dr)
    plt.plot(r, u, label="b = "+str(b))
plt.xlabel("r (fm)")
plt.ylabel("u(r)")
plt.legend()
plt.grid()
plt.show()