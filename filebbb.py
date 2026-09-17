import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

e_val = 3.795
hc =1973.0
m = 0.511*10**6
r_min = 0.001
r_max = 10.0
N = 2000
r = np.linspace(r_min,r_max,N)
dr = r[1]-r[0]
def V(r):
    return -(e_val**2)/r
factor = (hc**2)/(2.0*m)
diag = 2.0 / (dr**2)*np.ones(N)+V(r)/factor
off_diag = -1.0/(dr**2)*np.ones(N-1)
H_mat = np.diag(diag)+np.diag(off_diag,k=1)+np.diag(off_diag,k=-1)
evals,evecs=eigh(H_mat)
energies = evals *factor
E_ground = energies[0]
E_excited = energies[1]

print(f"Ground state energy:{E_ground: .3f} eV")
print(f"First excited state energy: {E_excited: .3f} eV")

psi_ground = evecs[:, 0]
psi_excited = evecs[:,1]
norm_ground = np.trapz(psi_ground**2,r)
norm_excited = np.trapz(psi_excited**2,r)
psi_ground_norm = psi_ground/np.sqrt(norm_ground)
psi_excited_norm = psi_excited/np.sqrt(norm_excited)
plt.figure(figsize=(10,12))
plt.plot(r,psi_ground_norm, label=f"Ground state energy:({E_ground: .3f} eV)")
plt.plot(r,psi_excited_norm, label=f"excited state energy:({E_excited: .3f} eV)")
plt.xlabel("r,(Å)")
plt.ylabel("Normalized Wavefunction ψ(r)")
plt.title("Normalized radial functions")
plt.grid(True)
plt.legend()
plt.show()