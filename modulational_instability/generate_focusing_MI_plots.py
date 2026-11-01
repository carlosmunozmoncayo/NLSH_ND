import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import sys

sys.path.append('../src/1d')  # Add the path to the traveling_waves directory
from nlsh1d import NLSH1D

# =====================================================================
# Exact NLSH plane-wave solution
# =====================================================================

def plane_wave(x, t, k, sigma, kappa, p, tau, branch: str = "+"):
    r"""Exact plane wave of the NLSH system.

        q0 = sigma exp(i(kx - omega t)),   q1 = i k/(1 + tau omega) q0,
        omega^{+-} = [ -(1 + a tau) +- sqrt(4 tau k^2 + (a tau - 1)^2) ]/(2 tau),
        a = kappa sigma^{p-1},

    i.e. (omega + a)(1 + tau omega) = k^2.  branch '+' -> omega -> k^2 - a as
    tau -> 0 (the NLS mode); branch '-' -> omega ~ -1/tau (the spurious fast
    relaxation mode, handy for stress-testing the implicit solver).

    Returns (q0, q1, omega).
    """
    a = kappa * sigma ** (p - 1.0)
    disc = np.sqrt(4.0 * tau * k * k + (a * tau - 1.0) ** 2)
    om = (-(1.0 + a * tau) + (1.0 if branch == "+" else -1.0) * disc) / (2.0 * tau)
    q0 = sigma * np.exp(1j * (k * x - om * t))
    q1 = (1j * k / (1.0 + tau * om)) * q0
    return q0, q1, om

def ab_alpha(kappa, sigma, p, tau):
    """Return (a, b, alpha) in the notation of the paper."""
    a = 1.0 - tau * kappa * sigma ** (p - 1.0)
    b = (p - 1.0) * kappa * sigma ** (p - 1.0) / 2.0
    return a, b, a + 2.0 * tau * b

def check_unstable_modes(k_perturbation, kappa, sigma, p, tau):
    a = 1-tau*kappa * sigma ** (p - 1.0)
    b = (p-1)*kappa*sigma**(p-1)/2
    k2 = np.abs(k_perturbation)**2
    D = (a**4+(4*tau*a**2+8*tau**2*a*b)*k2)
    print(f"Checking unstable modes for k={k_perturbation}, tau={tau}, D={D}")
    if D < 0:
        print(f"Unstable mode new band for k={k_perturbation}, tau={tau}, D={D}")
    print(f"Checking unstable modes for k={k_perturbation}, tau={tau}, k^2={k2}, 2ab={2*a*b}")
    if kappa>0 and k2< 2*a*b:
        print(f"Unstable mode classical band for k={k_perturbation}, tau={tau}, k^2 < 2ab={2*a*b}")

    #Compute roots of the dispersion relation (defaulting to NLS for tau->0)
    #Omega squared is either real or imaginary
    omega_p_sq = (a**2+2*tau*k2+np.sqrt(D+0.j))/(2*tau**2) 
    omega_m_sq = (a**2+2*tau*k2-np.sqrt(D+0.j))/(2*tau**2)
    print(f"omega_p_sq={omega_p_sq}, omega_m_sq={omega_m_sq}")
    omegas = [np.sqrt(omega_p_sq), -np.sqrt(omega_p_sq), np.sqrt(omega_m_sq), -np.sqrt(omega_m_sq)]
    arg_max = np.argmax(np.imag(omegas))
    return omegas[arg_max]

def unstable_eigenmode(x, k, kappa, sigma, p, tau, Om, amplitude):
    r"""Initial perturbation projected exactly onto the growing eigenmode.
 
    Solves the 2x2 sideband system for (eps_+, eps_-) and recovers
    delta_pm = i k eps_pm / (a +- tau Omega), then builds
 
        eps(x,0)   = eps_+ e^{ikx} + eps_-^* e^{-ikx}
        delta(x,0) = delta_+ e^{ikx} + delta_-^* e^{-ikx}.
 
    Seeding the eigenmode (rather than a bare cosine) removes the transient
    beating against the two stiff relaxation modes, so the measured growth
    rate matches Im(Omega) from t = 0 with no ad-hoc time shift.
    """
    a, b, _ = ab_alpha(kappa, sigma, p, tau)
    # row 1 of the sideband matrix:  (Om - k^2/(a + tau Om) + b) eps_+ + b eps_- = 0
    m11 = Om - k * k / (a + tau * Om) + b
    eps_p = 1.0 + 0j
    eps_m = -m11 / b * eps_p
    del_p = 1j * k * eps_p / (a + tau * Om)
    del_m = 1j * k * eps_m / (a - tau * Om)
    scale = amplitude / max(abs(eps_p), abs(eps_m))
    eps_p, eps_m, del_p, del_m = (z * scale for z in (eps_p, eps_m, del_p, del_m))
    e_ikx = np.exp(1j * k * x)
    eps = eps_p * e_ikx + np.conj(eps_m) / e_ikx
    delta = del_p * e_ikx + np.conj(del_m) / e_ikx
    return eps, delta


if __name__ == "__main__":
    L = 2*np.pi
    N_points = 2**10
    kappa = 1
    p = 3.0
    sigma = 1.0
    tau_list = [1.e-15,0.3, 0.4, 0.49, 0.51] #Less than 1.e-14 is going to default to NLS in the solver
    k = 0.
    Tfinal = 100.0
    dt = 1.e-2
    branch = '+'

    k_perturbation = 1

    r_list = []
    omega_unstable_list = []
    for tau in tau_list:
        print(f"Running simulation for tau={tau}")
        sol = NLSH1D(L=L, M=N_points, kappa=kappa, p=p, tau=tau)
        omega_unstable = check_unstable_modes(k_perturbation, kappa, sigma, p, tau)
        if tau < 1.e-10:
            omega_unstable = 1.j
        print(f"Unstable mode for tau={tau}: {omega_unstable}")
        omega_unstable_list.append(omega_unstable)
        q0i, q1i, om = plane_wave(sol.x, 0.0, k, sigma, kappa, p, tau, branch=branch)
        #Add perturbation
        eps_p = 1.e-3
        eps_m = 1.e-3
        if omega_unstable.imag > 1.e-10:
            d_eps, d_delta = unstable_eigenmode(sol.x, k_perturbation, kappa, sigma,
                                                p, tau, omega_unstable, eps_p)
            q0i = q0i + d_eps
            q1i = q1i + d_delta
        else:
            q0i = q0i + eps_p * np.cos(k_perturbation * sol.x)
        # q0i += eps_p * np.cos(k_perturbation * sol.x) 
        r = sol.run(q0i, q1i,Tfinal,dt, method="ars222", relaxation=True,n_snapshots=300)
        r_list.append(r)

    x = sol.x

    #First plot
    fs1 = 12   # ticks, and anything not called out below
    fs2 = 15   # axis labels + legend

    shifts = np.zeros_like(tau_list, dtype=float)
    # shifts[:4] = [0.5, 1., 3, 10]
    colors = ["#1ADF1A", '#FFC107', '#1E88E5', '#D81B60', '#FE6100']

    fig, ax = plt.subplots(figsize=(6, 6.5))

    lw = 2

    for idx, tau in enumerate(tau_list):
        r = r_list[idx]
        max_norm = [np.max(np.abs(q0)) for q0 in r.snapshots.q0]
        label_tau = r'$\tau$' + f'={tau}' if tau > 1.e-12 else 'NLS'
        ax.plot(r.snapshots.t, max_norm, label=label_tau, linewidth=lw, color=colors[idx])
        growth_rate = eps_p*np.exp(omega_unstable_list[idx].imag*(r.snapshots.t - shifts[idx]))
        ax.plot(r.snapshots.t, 1 + growth_rate, linestyle='dashed', color='black', linewidth=lw)

    ax.plot([], [], linestyle='dashed', color='black', linewidth=lw, label=r'$\Gamma_{\tau}(t)$')

    ax.set_xlim(0, 70)
    ax.set_ylim(0.9, 2.5)
    ax.set_xlabel('Time', fontsize=fs2)
    ax.set_ylabel(r'$\|q_0\|_\infty$', fontsize=fs2)
    ax.tick_params(axis='both', which='major', labelsize=fs1)

    n_entries = len(tau_list) + 1
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
            ncol=min(n_entries, 3), fontsize=fs2,
            frameon=False, columnspacing=1.2, handlelength=1.8)

    fig.subplots_adjust(top=0.85)   # room for the legend strip
    #Save as pdf
    plt.savefig('focusing_MI_threshold.pdf', bbox_inches='tight')


    #Second plot: contour plot of the solution for each tau
    fs1 = 12   # ticks
    fs2 = 15   # axis labels, titles, colorbar label

    x_scaled = sol.x / (2*np.pi)
    X, T = np.meshgrid(x_scaled, r.snapshots.t)

    idxs = [0, 1, 2, 3,4]
    Qs = [np.abs(r_list[i].snapshots.q0) for i in idxs]
    cmap = 'RdBu'

    # shared color scale
    vmin = 0.5
    vmax = 1.5
    levels = np.linspace(vmin, vmax, 21)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(1, 5, figsize=(6, 7), sharey=True,
                            gridspec_kw={'wspace': 0})


    for ax, i, Q in zip(axs, idxs, Qs):
        ax.pcolormesh(X, T, Q, norm=norm, cmap=cmap)
        ax.set_xlabel(r'$x/(2\pi)$', fontsize=fs2)
        ax.set_title(r'$\tau$=' + f'{tau_list[i]}' if tau_list[i] > 1.e-12 else 'NLS', fontsize=fs2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', which='major', labelsize=fs1)

    axs[0].set_ylabel('Time', fontsize=fs2)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cbar_ax, label=r'$|q_0|$',extend='both')
    cbar.set_label(r'$|q_0|$', fontsize=fs2)
    cbar.ax.tick_params(labelsize=fs1)

    plt.savefig('focusing_MI_contour.png', bbox_inches='tight', dpi=300)