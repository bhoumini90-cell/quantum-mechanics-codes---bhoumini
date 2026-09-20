import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
l=2
m=1
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
Y = sph_harm_y(l, m, theta, phi)
theta, phi = np.meshgrid(theta, phi)
P = abs(Y)**2
x = P * np.sin(theta) * np.cos(phi)
y = P * np.sin(theta) * np.sin(phi)
z = P * np.cos(theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)
ax.set_title(f"Spherical Harmonic |Y({l},{m})|²")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
hbar = 1
m = 1
V0 = 20
a = 1
def f(E):
    k = np.sqrt(2*m*(E+V0))/hbar
    alpha = np.sqrt(-2*m*E)/hbar
    return k*np.tan(k*a) - alpha
E = brentq(f, -19.9, -10.1)
print("Energy =", E)
x = np.linspace(-2*a, 2*a, 500)
k = np.sqrt(2*m*(E+V0))/hbar
alpha = np.sqrt(-2*m*E)/hbar
psi = np.zeros(len(x))
inside = abs(x) <= a
psi[inside] = np.cos(k*x[inside])
psi[x > a] = np.cos(k*a)*np.exp(-alpha*(x[x > a]-a))
psi[x < -a] = np.cos(k*a)*np.exp(alpha*(x[x < -a]+a))
dx = x[1]-x[0]
psi = psi/np.sqrt(np.sum(psi**2)*dx)
plt.plot(x, psi)
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Finite Potential Well")
plt.grid()
plt.show()