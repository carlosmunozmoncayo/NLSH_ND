import numpy as np
import matplotlib.pyplot as plt
from plane_wave_rectangle_nlsh import setup, plane_wave_NLSH_2D

#Same convergence test, only using the SharpClaw solver instead of the FV solver.
tau_list = [1, 0.1, 0.01, 0.001]

mx_list = [8,16,32, 64]

sigma = 1.0
pnonlinear = 3.0
kappa = -1.0
tfinal = 1.0

def L2_error_grid_function(dx, q, q_exact):
    """Compute the L2 error between the complex valued grid function q and the exact solution q_exact, given the grid spacing dx."""
    error = np.sqrt(np.sum(np.abs(q - q_exact)**2) * dx**2)
    return error

#Check if file with errors already exists
import os
if os.path.exists('convergence_sharpclaw_errors.pkl'):
    import pickle
    with open('convergence_sharpclaw_errors.pkl', 'rb') as f:
        errors = pickle.load(f)
else:
    errors = {}
    for tau in tau_list:
        errors[tau] = np.zeros((len(mx_list), 2))  # Store errors for q0 and q1
        for mx in mx_list:
            claw = setup(tau=tau, num_cells=(mx, mx), tfinal=tfinal, solver_type='sharpclaw',
                          pnonlinear=pnonlinear, kappa=kappa, sigma=sigma)
            claw.run()
            q = claw.frames[-1].q
            dx = claw.frames[-1].state.grid.delta[0]
            q0 = q[0, :, :] + 1j * q[3, :, :]
            q1 = np.zeros_like(q[:2, :, :], dtype=complex)
            q1[0, :, :] = q[1, :, :] + 1j * q[4, :, :]
            q1[1, :, :] = q[2, :, :] + 1j * q[5, :, :]
            X, Y = claw.frames[-1].state.grid.p_centers
            q0_real, q0_imag, q1_real, q1_imag, q2_real, q2_imag= plane_wave_NLSH_2D(X, Y, tfinal, kappa, sigma, np.sqrt(2), np.sqrt(2), tau=tau, p=pnonlinear, branch="physical")
            q0_exact = q0_real + 1j * q0_imag
            q1_exact = np.zeros_like(q1, dtype=complex)
            q1_exact[0, :, :] = q1_real + 1j * q1_imag
            q1_exact[1, :, :] = q2_real + 1j * q2_imag
            error_q0 = L2_error_grid_function(dx, q0, q0_exact)
            error_q1 = L2_error_grid_function(dx, q1, q1_exact)
            errors[tau][mx_list.index(mx), 0] = error_q0
            errors[tau][mx_list.index(mx), 1] = error_q1

    #Save errors to a file
    import pickle
    with open('convergence_sharpclaw_errors.pkl', 'wb') as f:
        pickle.dump(errors, f)

#plot convergence results
fig, ax = plt.subplots(1,2,figsize=(10, 4))
linestyle_q0 = 'o-'
linestyle_q1 = 's-'
#Dot dasheed lines for ref
linestyle_ref = 'k-.'
colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, tau in enumerate(tau_list):
    ax[0].plot(mx_list, errors[tau][:, 0], linestyle_q0, color=colors[i], label=rf'$\tau={tau}$')
    ax[1].plot(mx_list, errors[tau][:, 1], linestyle_q1, color=colors[i], label=rf'$\tau={tau}$')  
    
#Plot reference line for second order convergence aligned with the mean among taus
ref_line = [errors[tau_list[-1]][-1, 0] * (mx_list[-1] / mx)**5 for mx in mx_list]
ax[0].plot(mx_list, ref_line, linestyle_ref, color='k', label=r'$\mathcal{O}(\Delta x^5)$')

ref_line = [errors[tau_list[-1]][-1, 1] * (mx_list[-1] / mx)**5 for mx in mx_list]
ax[1].plot(mx_list, ref_line, linestyle_ref, color='k', label=r'$\mathcal{O}(\Delta x^5)$')

for a in ax:
    a.set_xscale('log', base=2)
    a.set_yscale('log')
    a.set_xlabel(r'$mx$')
    a.grid(True, which='both', alpha=0.3)
#SEt legend above the plots
ax[0].legend(loc='upper center', bbox_to_anchor=(1.125, 1.15), ncol=5, fontsize=12)
ax[0].set_ylabel(r'$||q_0^{\text{Num}}(T)-q_0^{\text{Ex}}(T)||_2$')
ax[1].set_ylabel(r'$||\mathbf{q}_1^{\text{Num}}(T)-\mathbf{q}_1^{\text{Ex}}(T)||_2$')
plt.savefig('convergence_sharpclaw_solver_NLSH.pdf')
plt.show()

#Linear interpolation on errors[tau][-1, 0] as a function of tau to see if it matches linear growth as tau decreases
plt.figure(figsize=(6, 4))
plt.loglog(tau_list, [errors[tau][-1, 0] for tau in tau_list], 'o-', label=r'$||q_0^{\text{Num}}(T)-q_0^{\text{Ex}}(T)||_2$')
#Reference line for 1/tau growth
ref_line = [errors[tau_list[-1]][-1, 0] * (tau_list[-1] / tau) for tau in tau_list]
plt.loglog(tau_list, ref_line, 'k--', label=r'$\mathcal{O}(1/\tau)$')
#REference line for 1/sqrt(tau) growth
ref_line = [errors[tau_list[-1]][-1, 0] * (tau_list[-1] / tau)**0.5 for tau in tau_list]
plt.loglog(tau_list, ref_line, 'k:', label=r'$\mathcal{O}(1/\sqrt{\tau})$')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$L^2$ error for $q_0$ at finest grid')
plt.title(r'Error for $q_0$ at finest grid as a function of $\tau$')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('convergence_sharpclaw_solver_NLSH_tau.pdf')
plt.show()
