"""
generate_focusing_MI_growth_plot.py
===================================

Growth-rate verification for the low-wavenumber (Benjamin-Feir) instability of
the focusing NLSH system.  Same experiment as `generate_focusing_MI_plots.py`,
but plotted as the deviation from the background,

    ||q0(., t)||_inf - sigma ,

on a log axis.  The linear theory then appears as a straight line of slope
max Im(Omega(tau)), drawn dashed in the same colour as the run it belongs to.
Each run is shown only up to its first maximum (the end of the linear regime);
the recurrence that follows is shown in the contour plot made by
`generate_focusing_MI_plots.py`.

Output: focusing_MI_growth.pdf
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '../src/1d'))
sys.path.append(HERE)
from nlsh1d import NLSH1D
from generate_focusing_MI_plots import (plane_wave, check_unstable_modes,
                                        unstable_eigenmode)
from plot_utils import growth_figure, keep_texts_inside

if __name__ == "__main__":
    L = 2*np.pi
    N_points = 2**10
    kappa = 1
    p = 3.0
    sigma = 1.0
    tau_list = [1.e-15, 0.3, 0.4, 0.49, 0.51]  # Less than 1.e-14 is going to default to NLS in the solver
    k = 0.
    Tfinal = 50.0
    dt = 1.e-2
    branch = '+'

    k_perturbation = 1
    eps_p = 1.e-3

    r_list = []
    omega_unstable_list = []
    for tau in tau_list:
        print(f"Running simulation for tau={tau}")
        sol = NLSH1D(L=L, M=N_points, kappa=kappa, p=p, tau=tau)
        omega_unstable = check_unstable_modes(k_perturbation, kappa, sigma, p, tau)
        if tau < 1.e-10:
            omega_unstable = 1.j
        omega_unstable_list.append(omega_unstable)
        q0i, q1i, om = plane_wave(sol.x, 0.0, k, sigma, kappa, p, tau, branch=branch)
        if omega_unstable.imag > 1.e-10:
            d_eps, d_delta = unstable_eigenmode(sol.x, k_perturbation, kappa, sigma,
                                                p, tau, omega_unstable, eps_p)
            q0i = q0i + d_eps
            q1i = q1i + d_delta
        else:
            q0i = q0i + eps_p * np.cos(k_perturbation * sol.x)
        r = sol.run(q0i, q1i, Tfinal, dt, method="ars443", relaxation=True,
                    n_snapshots=500)
        r_list.append(r)

    fs1 = 12   # ticks, and anything not called out below
    fs2 = 15   # axis labels, legend, line labels
    lw = 2

    # ordered colours: tau increases from NLS (tau -> 0) to the stable tau = 0.51
    colors = plt.cm.viridis(np.linspace(0.0, 0.82, len(tau_list)))

    # fig, ax = plt.subplots(figsize=(6, 6.75))
    fig, ax = growth_figure()

    for idx, tau in enumerate(tau_list):
        r = r_list[idx]
        t = np.asarray(r.snapshots.t)
        dev = np.array([np.max(np.abs(q0)) for q0 in r.snapshots.q0]) - sigma
        growth = omega_unstable_list[idx].imag
        label_tau = r'$\tau$' + f'={tau}' if tau > 1.e-12 else 'NLS'

        if growth > 1.e-10:
            # first local maximum once the perturbation is O(0.1)
            i_peak = np.flatnonzero((dev[:-1] > dev[1:]) & (dev[:-1] > 0.05))[0]
            # theory drawn a little past the saturation level of its run
            t_end = np.log(1.3 * dev[i_peak] / dev[0]) / growth
            label_xy, ha = (t_end + 0.6, 1.3 * dev[i_peak]), 'left'
        else:
            i_peak = len(t) - 1
            t_end = Tfinal
            label_xy, ha = (Tfinal - 0.5, 1.4 * dev[0]), 'right'

        ax.semilogy(t[:i_peak + 1], dev[:i_peak + 1], color=colors[idx], linewidth=lw)
        t_th = np.linspace(0, t_end, 200)
        ax.semilogy(t_th, dev[0] * np.exp(growth * t_th), color=colors[idx],
                    linewidth=0.75*lw, linestyle=(0, (3, 2)))
        ax.text(*label_xy, label_tau, color=colors[idx], fontsize=fs2,
                ha=ha, va='bottom')

    ax.plot([], [], color='black', linewidth=lw, label='simulation')
    ax.plot([], [], color='black', linewidth=0.75*lw, linestyle=(0, (3, 2)),
            label='linear theory')

    ax.set_xlim(0, Tfinal)
    ax.set_ylim(4.e-4, 3.)
    ax.set_xlabel('Time', fontsize=fs2)
    ax.set_ylabel(r'$\|q_0\|_\infty - \sigma$', fontsize=fs2)
    ax.tick_params(axis='both', which='major', labelsize=fs1)
    ax.legend(loc='center right', fontsize=fs2, frameon=False, handlelength=2.2)
    keep_texts_inside(ax)

    plt.savefig(os.path.join(HERE, '../paper/figures/focusing_MI_growth.pdf'))
