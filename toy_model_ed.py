#!/usr/bin/env python3
"""
Exact diagonalization of a number of 1D toy models (for now just V(x) = k (x^2 - c0)^2).
Outputs energy levels, 0->1 gap and a thermal (dipole-weighted) average frequency at T=20 K.

Parameters to edit:
Spring constant, k (Ha/Bohr^4) 
SDW well centers:c0 (Bohr^2)
pot = "sdw" (symmetric double well), "harmonic", "morse"
N = number of grid points

"""

import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Exact diagonalization of selected potential.")
parser.add_argument("--k", type=float, default=0.183736, help="Potential parameter k (Ha/Bohr^4)")
parser.add_argument("--c0", type=float, default=0.3, help="Potential parameter c0 (Bohr^2)")
parser.add_argument("--pot", type=str, default="sdw", choices=["sdw", "harmonic", "morse"])
parser.add_argument("--N", type=int, default=2000, help="Number of grid points (N) for discretization")
args= parser.parse_args()
HA_to_cm = 219_474.6313705          # 1 Hartree in cm^-1
kB_Ha_per_K = 3.166811563e-6        # k_B in Hartree/K
amu_to_me = 1822.888486209          # atomic mass unit in electron masses (a.u.)


k   = 0.183736        # Hartree / Bohr^4
c0  = args.c0            # Bohr^2
T_K = 20.0            # temperature in Kelvin

# Discretization
N   = args.N      # grid points

# Choose box size automatically from well separation and barrier height
x0 = np.sqrt(max(c0, 1e-12))
L_pad = max(6*x0, 20.0*np.power(max(k*c0**2, 1e-9), -0.125))  # heuristic padding
xmax = L_pad
xmin = -L_pad

# Build grid and operators 
x = np.linspace(xmin, xmax, N)
dx = x[1] - x[0]

# Mass in atomic units (electron masses)
m = 1837.15417302

# Potential
V = k * (x**2 - c0)**2

# Kinetic energy via 2nd-order finite differences 
# T = - (1/(2m)) d^2/dx^2
diag = np.full(N, 1.0/dx**2)
off  = np.full(N-1, -0.5/dx**2) * 2.0   
# Discrete Laplacian matrix (tridiagonal)
H = np.zeros((N, N))
np.fill_diagonal(H, -2.0*diag)
np.fill_diagonal(H[1:], off)
np.fill_diagonal(H[:,1:], off)
# Kinetic prefactor
H *= -(1.0/(2.0*m))
# Add potential
H += np.diag(V)

# Diagonalize (lowest few states)
# Try sparse if available; fall back to dense
evals = None
evecs = None
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    Hs = sp.diags([np.diag(H, -1), np.diag(H, 0), np.diag(H, 1)], [-1, 0, 1], format="csc")
    n_states = 12  # compute a few low-lying states
    evals, evecs = spla.eigsh(Hs, k=n_states, which='SA')
    idx = np.argsort(evals)
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
except Exception:
    # Dense fallback
    evals, evecs = np.linalg.eigh(H)
    n_states = 12
    evals = evals[:n_states]
    evecs = evecs[:, :n_states]

# Normalize eigenvectors (finite-volume normalization)
for j in range(evecs.shape[1]):
    norm = np.sqrt(np.trapz(np.abs(evecs[:, j])**2, x))
    evecs[:, j] /= norm

# ----------- Frequencies and thermal weights --------------------------
# Energies in Hartree; transition angular frequencies are (E_m - E_n)/ħ (ħ=1 in a.u.)
# Spectral frequencies nu = (E_m - E_n) in cm^-1
# Fundamental 0->1:
Psi=evecs
for j in range(Psi.shape[1]):
    norm = np.sqrt(np.trapz(np.abs(Psi[:, j])**2, x))
    Psi[:, j] /= norm

E = evals


for j in range(Psi.shape[1]):
    norm = np.sqrt(np.trapz(np.abs(Psi[:, j])**2, x))
    Psi[:, j] /= norm

# ------------------ Frequencies / Splittings ------------------
# Energies are in Hartree; we report frequencies in cm^-1
# Tunnelling split Delta = E1 - E0
Delta_cm1 = (E[1] - E[0]) * HA_to_cm

# Intra-well vibrational frequency via doublet-center spacing:
# omega_vib ≈ [(E2+E3)/2 - (E0+E1)/2] 
if len(E) >= 4:
    omega_vib_cm1 = (((E[2] + E[3]) * 0.5) - ((E[0] + E[1]) * 0.5)) * HA_to_cm
else:
    omega_vib_cm1 = np.nan  # not enough states to form two doublets

# Fundamental 0→1 line (same as Delta for perfectly symmetric double well)
nu01_cm1 = (E[1] - E[0]) * HA_to_cm

# ------------------ Thermal & dipole-weighted averages ------------------
beta = 1.0 / (kB_Ha_per_K * T_K)
weights = np.exp(-(E - E[0]) * beta)
p = weights / np.sum(weights)

# Dipole operator ~ x; line strengths S_{n->m} = p_n |<n|x|m>|^2
def x_overlap(n, m):
    return np.trapz(Psi[:, n] * x * Psi[:, m], x)

x_mat = np.zeros((Psi.shape[1], Psi.shape[1]))
for n in range(Psi.shape[1]):
    for m2 in range(Psi.shape[1]):
        x_mat[n, m2] = x_overlap(n, m2)

lines_freq = []
lines_strength = []
for n in range(Psi.shape[1]):
    for m2 in range(n+1, Psi.shape[1]):
        nu_cm1 = (E[m2] - E[n]) * HA_to_cm
        S = p[n] * (x_mat[n, m2]**2)
        if nu_cm1 > 0 and S > 1e-14:
            lines_freq.append(nu_cm1)
            lines_strength.append(S)

lines_freq = np.array(lines_freq)
lines_strength = np.array(lines_strength)
nu_thermal_cm1 = (np.sum(lines_strength * lines_freq) / np.sum(lines_strength)
                  if lines_strength.size > 0 else np.nan)

# A simple Boltzmann-weighted nearest-neighbor spacing average 
nuT_cm1 = (np.sum((E[1:] - E[:-1]) * p[:-1]) * HA_to_cm) if len(E) > 1 else np.nan

# ------------------ Output ------------------
print("=== Exact Diagonalization for V(x) = k (x^2 - c0)^2 ===")
print(f"k = {k:.6g} Ha/Bohr^4, c0 = {c0:.6g} Bohr^2, mass = {mass_amu:.6g} amu, T = {T_K:.2f} K")
print(f"Grid: N={N}, x in [{xmin:.3f}, {xmax:.3f}] Bohr, dx={dx:.5f} Bohr")
print("Lowest energies (Ha): " + ", ".join(f"{Ei:.6e}" for Ei in E[:6]))

print("\n--- Key Spectral Quantities (cm^-1) ---")
print(f"Tunnelling split Δ = E1 - E0:         {Delta_cm1:.6f} cm^-1")
print(f"Intra-well ω_vib (doublet centers):   {omega_vib_cm1:.6f} cm^-1")
print(f"Fundamental 0→1 (for reference):      {nu01_cm1:.6f} cm^-1")

print("\n--- Thermal diagnostics ---")
print(f"Boltzmann-weighted avg spacing ν_T:    {nuT_cm1:.6f} cm^-1  at T={T_K:.1f} K")
if np.isfinite(nu_thermal_cm1):
    print(f"Dipole-weighted avg frequency:         {nu_thermal_cm1:.6f} cm^-1  at T={T_K:.1f} K")

if lines_strength.size:
    print("\nThermally weighted dipole-allowed lines (n→m):")
    for nu, S in sorted(zip(lines_freq, lines_strength), key=lambda t: t[0]):
        print(f"  {nu:12.6f} cm^-1   S = {S:.3e}")









