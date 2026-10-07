r"""
Generates ground state solutions to NLSH with Petviashvili iterations
and checks convergence agains NLS ground states.
No time evolution. Intended to verify the proposition 
of the Special Solutions section of the NLSH in ND paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '2d/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '3d/'))
from petviashvili import run_and_save as run_and_save2d
from petviashvili3d import run_and_save as run_and_save3d


#Fontsizes
fs1 = 16
fs2 = 14

plt.rcParams.update({'font.size': fs1})

#First in 2D
mu = 1.0
kappa = 1.0
tau_list = [10**(-i) for i in range(1, 6)] 
tau_list = [0.9] + tau_list

tau_NLS = 0.
#Generate an NLS one first
params_NLS, X_NLS, Y_NLS, q0_NLS, q1_NLS = run_and_save2d(tau=tau_NLS, mu=mu, kappa=kappa, save=False, niters=10)

error_list_q0 = []
error_list_q1 = []
for tau in tau_list:
    params, X, Y, q0, q1 = run_and_save2d(tau=tau, mu=mu, kappa=kappa, save=False, niters=10)
    error = np.max(np.abs(q0 - q0_NLS))
    error_list_q0.append(error)
    error = np.max(np.abs(q1 - q1_NLS))
    error_list_q1.append(error)

#Now in 3D
error_list_q0_3d = []
error_list_q1_3d = []
#Generate an NLS one first
params_NLS_3d, X_NLS_3d, Y_NLS_3d, Z_NLS_3d, q0_NLS_3d, q1_NLS_3d = run_and_save3d(tau=tau_NLS, mu=mu, kappa=kappa, save=False, niters=10)
for tau in tau_list:
    params_3d, X_3d, Y_3d, Z_3d, q0_3d, q1_3d = run_and_save3d(tau=tau, mu=mu, kappa=kappa, save=False, niters=10)
    error = np.max(np.abs(q0_3d - q0_NLS_3d))
    error_list_q0_3d.append(error)
    error = np.max(np.abs(q1_3d - q1_NLS_3d))
    error_list_q1_3d.append(error)

#Convergence plots
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(tau_list, error_list_q0, 'o-', label=r'$||\bar{q}_0-\bar{u}||_\infty$ 2D')
ax.loglog(tau_list, error_list_q1, 'o-', label=r'$||\bar{q}_1-\nabla\bar{u}||_\infty$ 2D')
ax.loglog(tau_list, error_list_q0_3d, 's-', label=r'$||\bar{q}_0-\bar{u}||_\infty$ 3D')
ax.loglog(tau_list, error_list_q1_3d, 's-', label=r'$||\bar{q}_1-\nabla\bar{u}||_\infty$ 3D')
#Plot reference line for tau^1 convergence
ref_line = [tau**1 for tau in tau_list]
ax.loglog(tau_list, ref_line, 'k--', label=r'$\mathcal{O}(\tau)$')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$L^\infty$ error')
fig.suptitle('Convergence of NLSH to NLS ground state as'+r' $\tau \to 0$')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.savefig('convergence_ground_state_NLSH_to_NLS.pdf')
