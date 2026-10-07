import numpy as np
import matplotlib.pyplot as plt
from plane_wave_rectangle_nlsh import setup, plane_wave_NLSH_2D

#Convergence test for the FV solver for the NLSH in 2D, using the plane wave solution as a reference. 
#The test is performed for different values of tau and grid resolutions, and the errors are computed in L2 norm for q0 and q1, as well as the mass drift. 
#The results are plotted to show convergence with respect to grid refinement.

tau_list = [1, 0.1, 0.01, 0.001]

mx_list = [32, 64, 128, 256]


sigma = 1.0
pnonlinear = 3.0
kappa = -1.0
tfinal = 1.0
k_wavenumber = 1#np.sqrt(2)
L_diagonal = 4*np.pi/k_wavenumber
L_x = L_diagonal/np.sqrt(2)
L_y = L_diagonal/np.sqrt(2)

dx_list = [L_x/mx for mx in mx_list]


def L2_error_grid_function(dx, q, q_exact):
    """Compute the L2 error between the complex valued grid function q and the exact solution q_exact, given the grid spacing dx."""
    error = np.sqrt(np.sum(np.abs(q - q_exact)**2) * dx**2)
    return error
def mass_func(q0, q1, dx,tau):
    """Compute the mass drift of the solution q."""
    mass = np.sum(np.abs(q0)**2)+tau*np.sum(np.abs(q1[0])**2+np.abs(q1[1])**2)
    mass = mass*dx**2
    return mass
def mass_drift(q0, q1, q0_exact, q1_exact, dx, tau):
    """Compute the mass drift of the solution q compared to the exact solution q_exact."""
    mass_q = mass_func(q0,q1, dx, tau)
    mass_q_exact = mass_func(q0_exact,q1_exact, dx, tau)
    drift = np.abs(mass_q - mass_q_exact)
    return drift

def mass_temporal_drift(claw,tau):
    mass_list = []
    for frame in claw.frames:
        q = frame.q
        dx = frame.state.grid.delta[0]
        q0 = q[0, :, :] + 1j * q[3, :, :]
        q1 = np.zeros_like(q[:2, :, :], dtype=complex)
        q1[0, :, :] = q[1, :, :] + 1j * q[4, :, :]
        q1[1, :, :] = q[2, :, :] + 1j * q[5, :, :]
        mass_list.append(mass_func(q0,q1, dx, tau))
    mass_list = np.array(mass_list)
    mass_drift = np.abs(mass_list - mass_list[0])
    return mass_drift

#Check if file with errors already exists
import os
if os.path.exists('convergence_errors.pkl') and os.path.exists('mass_temporal_drift.pkl'):
    import pickle
    with open('convergence_errors.pkl', 'rb') as f:
        errors = pickle.load(f)
    with open('mass_temporal_drift.pkl', 'rb') as f:
        mass_temporal_drift_dict = pickle.load(f)
else:
    errors = {}
    mass_temporal_drift_dict = {}
    for tau in tau_list:
        errors[tau] = np.zeros((len(mx_list), 3))  # Store errors for q0, q1, and mass drift
        for mx in mx_list:
            claw = setup(tau=tau, num_cells=(mx, mx), tfinal=tfinal, pnonlinear=pnonlinear, kappa=kappa, sigma=sigma)
            claw.run()
            q = claw.frames[-1].q
            dx = claw.frames[-1].state.grid.delta[0]
            q0 = q[0, :, :] + 1j * q[3, :, :]
            q1 = np.zeros_like(q[:2, :, :], dtype=complex)
            q1[0, :, :] = q[1, :, :] + 1j * q[4, :, :]
            q1[1, :, :] = q[2, :, :] + 1j * q[5, :, :]
            X, Y = claw.frames[-1].state.grid.p_centers
            q0_real, q0_imag, q1_real, q1_imag, q2_real, q2_imag= plane_wave_NLSH_2D(X, Y, tfinal, kappa, sigma, k_wavenumber/np.sqrt(2), k_wavenumber/np.sqrt(2), tau=tau, p=pnonlinear, branch="physical")
            q0_exact = q0_real + 1j * q0_imag
            q1_exact = np.zeros_like(q1, dtype=complex)
            q1_exact[0, :, :] = q1_real + 1j * q1_imag
            q1_exact[1, :, :] = q2_real + 1j * q2_imag
            error_q0 = L2_error_grid_function(dx, q0, q0_exact)
            error_q1 = L2_error_grid_function(dx, q1, q1_exact)
            error_mass_drift = mass_drift(q0, q1, q0_exact, q1_exact, dx, tau)
            errors[tau][mx_list.index(mx), 0] = error_q0
            errors[tau][mx_list.index(mx), 1] = error_q1
            errors[tau][mx_list.index(mx), 2] = error_mass_drift
        #Save mass temporal drift for finest grid for each tau
        mass_temporal_drift_dict[tau] = mass_temporal_drift(claw, tau)
    #Save errors to a file
    import pickle
    with open('convergence_errors.pkl', 'wb') as f:
        pickle.dump(errors, f)
    with open('mass_temporal_drift.pkl', 'wb') as f:
        pickle.dump(mass_temporal_drift_dict, f)

#plot convergence results
fs = 18
fs2 = 16
plt.rcParams.update({'font.size': fs, 'axes.labelsize': fs2, 'axes.titlesize': fs2, 'legend.fontsize': fs2, 'xtick.labelsize': fs,
                     'ytick.labelsize': fs})

fig, ax = plt.subplots(1,3,figsize=(24, 6))
linestyle_q0 = 'o-'
linestyle_q1 = 's-'
#Dot dasheed lines for ref
linestyle_ref1 = 'k--'
linestyle_ref2 = 'k-.'
linestyle_ref3 = 'k:'
colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
times = np.linspace(0, tfinal, len(mass_temporal_drift_dict[tau_list[0]]))
for i, tau in enumerate(tau_list):
    ax[0].plot(dx_list, errors[tau][:, 0], linestyle_q0, color=colors[i], label=rf'$\tau={tau}$')
    ax[1].plot(times, mass_temporal_drift_dict[tau], linestyle_q1, color=colors[i], label=rf'$\tau={tau}$')
    ax[2].plot(dx_list, errors[tau][:, 2], linestyle_q1, color=colors[i], label=rf'$\tau={tau}$')  


#REference lines for second and third order of convergence
ref_line = [errors[tau_list[1]][-1, 0] * (mx_list[-1] / mx)**2 for mx in mx_list]
ref_line2 = [errors[tau_list[2]][-1, 0] * (mx_list[-1] / mx)**2 for mx in mx_list]
ref_line = [(r1+r2)/2 for r1, r2 in zip(ref_line, ref_line2)]
ax[0].plot(dx_list, ref_line, linestyle_ref2, color='k', label=r'$\mathcal{O}(\Delta x^2)$')

#Mass drift (temporal) with finest grid for each tau (linear reference line O(t))
ref_line = [mass_temporal_drift_dict[tau_list[1]][-1] * (t / times[-1]) for t in times]
ax[1].plot(times, ref_line, linestyle_ref1, color='k', label=r'$\mathcal{O}(t)$')


#Mass drift with grid refinement
ref_line = [errors[tau_list[1]][-1, 2] * (mx_list[-1] / mx)**2 for mx in mx_list]
ax[2].plot([], [], linestyle_ref1, color='k', label=r'$\mathcal{O}(t)$')
ax[2].plot(dx_list, ref_line, linestyle_ref2, color='k', label=r'$\mathcal{O}(\Delta x^2)$')
ref_line = [errors[tau_list[1]][-1, 2] * (mx_list[-1] / mx)**3 for mx in mx_list]
ax[2].plot(dx_list, ref_line, linestyle_ref3, color='k', label=r'$\mathcal{O}(\Delta x^3)$')


for a in ax:
    a.set_xscale('log', base=2)
    a.set_yscale('log')
    a.set_xlabel(r'$\Delta x$')
    a.grid(True, which='both', alpha=0.3)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$t$')

#SEt legend above the plots
ax[2].legend(loc='upper center', bbox_to_anchor=(-0.75, 1.15), ncol=7, fontsize=fs2)
ax[0].set_ylabel(r'$||q_0^{\text{Num}}(T)-q_0^{\text{Ex}}(T)||_2$')
ax[1].set_ylabel(r'$|\bar{M}(t)-\bar{M}(0)|$ (finest grid)')
ax[2].set_ylabel(r'$|\bar{M}(T)-\bar{M}(0)|$')
plt.savefig('convergence_FV_solver_NLSH.pdf',
            bbox_inches='tight', pad_inches=0.02)
plt.show()

#Linear interpolation on errors[tau][-1, 0] as a function of tau to see if it matches linear growth as tau decreases
plt.figure(figsize=(6, 4))
plt.loglog(tau_list, [errors[tau][-1, 0] for tau in tau_list], 'o-', label=r'$||q_0^{\text{Num}}(T)-q_0^{\text{Ex}}(T)||_2$')
#Reference line for 1/tau growth
ref_line = [errors[tau_list[-1]][-1, 0] * (tau_list[-1] / tau) for tau in tau_list]
plt.loglog(tau_list, ref_line, 'k--', label=r'$\mathcal{O}(1/\tau)$')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$L^2$ error for $q_0$ at finest grid')
plt.title(r'Error for $q_0$ at finest grid as a function of $\tau$')
plt.legend()
plt.grid()
plt.savefig('convergence_FV_solver_NLSH_tau.pdf')
plt.show()

