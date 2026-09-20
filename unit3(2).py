import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
e = 3.795
m = 0.511e6
hbar_c = 1973
K = hbar_c**2 / (2*m)
rmax = 50
N = 3000
dr = rmax / N
r = np.arange(1, N) * dr
a_values = [3, 5, 7]
for a in a_values:
    V = -e**2 * np.exp(-r/a) / r
    main = 2*K/dr**2 + V
    side = -K/dr**2
    H = diags(
        [side*np.ones(N-2), main, side*np.ones(N-2)],
        [-1, 0, 1]
    )
    energy, wave = eigsh(H, k=1, which='SA')
    E0 = energy[0]
    u = wave[:, 0]
    u = u / np.sqrt(np.sum(u**2) * dr)
    print("a =", a, "Å")
    print("Ground state energy =", round(E0, 3), "eV")
    print()
    plt.plot(r, u, label="a = " + str(a) + " Å")
plt.xlabel("r (Å)")
plt.ylabel("u(r)")
plt.title("Ground State Wavefunction")
plt.legend()
plt.grid(True)
plt.xlim(0, 20)
plt.show()