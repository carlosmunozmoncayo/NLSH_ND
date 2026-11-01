"""
generate_defocusing_MI_growth_plot.py
=====================================

Growth-rate verification for the high-wavenumber (D < 0) instability band of
the defocusing NLSH system.  Same experiment as
`generate_defocusing_MI_plots.py`, but plotted as the deviation from the
background,

    ||q0(., t)||_inf - sigma ,

on a log axis.  The linear theory then appears as a straight line of slope
Im(Omega), drawn dashed in the same colour as the run it belongs to.  Re(Omega)
does not enter: the growing mode is a travelling wave, and the maximum over x
of a travelling wave does not depend on where the wave is.

Each unstable run is shown only up to its first maximum.  Unlike the focusing
case there is no recurrence after it: the harmonics 2k, 3k, ... lie in the
band as well, so ||q0||_inf runs ahead of the theory line just before the
maximum and the energy then cascades to the grid scale (see the contour plot
made by `generate_defocusing_MI_plots.py`).

The stable runs (NLS, and tau = 0.9 < tau_band) are seeded with a *neutral*
eigenmode rather than the bare cosine used there. 

The theory lines start from the measured deviation at t = 0.  To first order
this is |eps_+ + eps_-|, which depends on the (complex) eigenvector and can lie
anywhere between 0 and 2 eps.

Output: ../paper/figures/defocusing_MI_growth.pdf
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '../src/1d'))
sys.path.append(HERE)
from nlsh1d import NLSH1D
from generate_defocusing_MI_plots import (plane_wave, ab_alpha, report,
                                          eigenmode_perturbation)
from plot_utils import growth_figure, keep_texts_inside


def neutral_mode(k, kappa, sigma, p, tau):
    r"""Physical real root Omega of a modulationally stable sideband k.

    This is sqrt(om_m_sq) from `dispersion_roots`, rewritten with
    om_p_sq * om_m_sq = k^2 (k^2 - 2ab) / tau^2 so that it does not cancel
    catastrophically as tau -> 0, where it tends to the NLS value
    k sqrt(k^2 - 2b).  Valid only when k is stable (D >= 0, k^2 >= 2ab).
    """
    a, b, _ = ab_alpha(kappa, sigma, p, tau)
    k2 = k * k
    D = a**4 + (4.0*tau*a**2 + 8.0*tau**2*a*b) * k2
    return complex(np.sqrt(2.0*k2*(k2 - 2.0*a*b) / (a**2 + 2.0*tau*k2 + np.sqrt(D))))


if __name__ == "__main__":
    L = 2*np.pi
    N_points = 2**10
    kappa = -1.0          # DEFOCUSING
    p = 3.0
    sigma = 1.0
    tau_list = [1.e-15, 0.9, 1.1, 1.5, 10.0]  # Less than 1.e-14 is going to default to NLS in the solver
    k = 0.
    Tfinal = 50.0         # tau = 1.1 (Im Omega ~ 0.17) needs ~40 to saturate
    dt = 5.e-3
    branch = '+'

    k_perturbation = 8    # must be an integer multiple of 2 pi / L
    eps_p = 1.e-3

    r_list = []
    growth_list = []
    for tau in tau_list:
        print(f"Running simulation for tau={tau}")
        sol = NLSH1D(L=L, M=N_points, kappa=kappa, p=p, tau=tau)
        if tau < 1.e-10:
            # defocusing NLS is modulationally stable (and dispersion_roots
            # cancels catastrophically at this tau)
            Om = neutral_mode(k_perturbation, kappa, sigma, p, tau)
        else:
            Om = report(k_perturbation, kappa, sigma, p, tau)
            if Om.imag <= 1.e-10:
                Om = neutral_mode(k_perturbation, kappa, sigma, p, tau)
        growth_list.append(Om.imag)
        q0i, q1i, om = plane_wave(sol.x, 0.0, k, sigma, kappa, p, tau, branch=branch)
        # unstable: the growing eigenmode; stable: a neutral one (see docstring)
        d_eps, d_delta = eigenmode_perturbation(sol.x, k_perturbation, kappa, sigma,
                                                p, tau, Om, eps_p)
        r = sol.run(q0i + d_eps, q1i + d_delta, Tfinal, dt, method="ars443",
                    relaxation=True, n_snapshots=600)
        r_list.append(r)

    fs1 = 12   # ticks, and anything not called out below
    fs2 = 15   # axis labels, legend, line labels
    lw = 2
    y_lim = (4.e-4, 3.)

    # ordered colours: tau increases from NLS (tau -> 0) through the stable
    # tau = 0.9 to the strongly unstable tau = 10
    colors = plt.cm.viridis(np.linspace(0.0, 0.82, len(tau_list)))

    devs = [np.array([np.max(np.abs(q0)) for q0 in r.snapshots.q0]) - sigma
            for r in r_list]
    unstable = [g > 1.e-10 for g in growth_list]
    # Label placement.  The two steepest lines (tau = 1.5, 10) are close, so the
    # steepest is labelled on its left, where the plot is empty.  The two flat
    # lines are close too, so the lower one is labelled underneath.
    i_steep = int(np.argmax(growth_list))
    i_low = min((i for i in range(len(tau_list)) if not unstable[i]),
                key=lambda i: devs[i][0], default=None)

    # fig, ax = plt.subplots(figsize=(6, 6.75))
    fig, ax = growth_figure()


    for idx, tau in enumerate(tau_list):
        t = np.asarray(r_list[idx].snapshots.t)
        dev = devs[idx]
        growth = growth_list[idx]
        label_tau = r'$\tau$' + f'={tau}' if tau > 1.e-12 else 'NLS'

        if unstable[idx]:
            # first local maximum once the perturbation is O(0.1); fall back to
            # the end of the run if it has not saturated by Tfinal
            peaks = np.flatnonzero((dev[:-1] > dev[1:]) & (dev[:-1] > 0.05))
            i_peak = peaks[0] if peaks.size else len(t) - 1
            # theory drawn a little past the saturation level of its run
            t_end = np.log(1.3 * dev[i_peak] / dev[0]) / growth
            # label next to the theory line at that level, kept inside the axes
            # (the cascade has no breather peak to bound ||q0||_inf)
            y_lab = min(1.3 * dev[i_peak], y_lim[1] / 1.5)
            t_lab = np.log(y_lab / dev[0]) / growth
            if idx == i_steep:
                label_xy, ha, va = (t_lab - 0.6, y_lab), 'right', 'bottom'
            else:
                label_xy, ha, va = (t_lab + 0.9, y_lab-0.3), 'left', 'bottom'
        else:
            i_peak = len(t) - 1
            t_end = Tfinal
            if idx == i_low:
                label_xy, ha, va = (Tfinal - 0.5, dev[0] / 1.4), 'right', 'top'
            else:
                label_xy, ha, va = (Tfinal - 0.5, 1.4 * dev[0]), 'right', 'bottom'

        ax.semilogy(t[:i_peak + 1], dev[:i_peak + 1], color=colors[idx], linewidth=lw)
        t_th = np.linspace(0, t_end, 200)
        ax.semilogy(t_th, dev[0] * np.exp(growth * t_th), color=colors[idx],
                    linewidth=0.75*lw, linestyle=(0, (3, 2)))
        ax.text(*label_xy, label_tau, color=colors[idx], fontsize=fs2,
                ha=ha, va=va)

    ax.plot([], [], color='black', linewidth=lw, label='simulation')
    ax.plot([], [], color='black', linewidth=0.75*lw, linestyle=(0, (3, 2)),
            label='linear theory')

    ax.set_xlim(0, Tfinal)
    ax.set_ylim(*y_lim)
    ax.set_xlabel('Time', fontsize=fs2)
    ax.set_ylabel(r'$\|q_0\|_\infty - \sigma$', fontsize=fs2)
    ax.tick_params(axis='both', which='major', labelsize=fs1)
    ax.legend(loc='center right', fontsize=fs2, frameon=False, handlelength=2.2)
    keep_texts_inside(ax)

    plt.savefig(os.path.join(HERE, 'defocusing_MI_growth.pdf'))