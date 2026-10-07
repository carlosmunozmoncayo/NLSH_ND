"""
generate_defocusing_MI_plots.py
===============================

Numerical verification of the *new* high-wavenumber instability band of the
NLSH system, i.e. the mechanism  D < 0  of Section \ref{sec:MI theoretical}.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import sys

sys.path.append('../src/1d')
from nlsh1d import NLSH1D


# =====================================================================
# Exact NLSH plane-wave solution  (identical to the focusing script)
# =====================================================================

def plane_wave(x, t, k, sigma, kappa, p, tau, branch: str = "+"):
    r"""Exact plane wave of the NLSH system.

        q0 = sigma exp(i(kx - omega t)),   q1 = i k/(1 + tau omega) q0,
        (omega + a)(1 + tau omega) = k^2,   a = kappa sigma^{p-1}.

    Branch '+' -> omega -> k^2 - a as tau -> 0 (the NLS mode).  In the
    defocusing case 1 - tau kappa sigma^{p-1} > 0 always, so the '+' label
    never swaps meaning (it can for kappa > 0 and tau kappa sigma^{p-1} > 1).

    Returns (q0, q1, omega).
    """
    a = kappa * sigma ** (p - 1.0)
    disc = np.sqrt(4.0 * tau * k * k + (a * tau - 1.0) ** 2)
    om = (-(1.0 + a * tau) + (1.0 if branch == "+" else -1.0) * disc) / (2.0 * tau)
    q0 = sigma * np.exp(1j * (k * x - om * t))
    q1 = (1j * k / (1.0 + tau * om)) * q0
    return q0, q1, om


# =====================================================================
# Linear modulational stability of the k = 0 plane wave
# =====================================================================

def ab_alpha(kappa, sigma, p, tau):
    """Return (a, b, alpha) in the notation of the paper."""
    a = 1.0 - tau * kappa * sigma ** (p - 1.0)
    b = (p - 1.0) * kappa * sigma ** (p - 1.0) / 2.0
    return a, b, a + 2.0 * tau * b


def dispersion_roots(k_perturbation, kappa, sigma, p, tau):
    """All four roots Omega of  tau^2 Om^4 - (a^2 + 2 tau k^2) Om^2 + k^2(k^2 - 2ab) = 0."""
    a, b, _ = ab_alpha(kappa, sigma, p, tau)
    k2 = np.abs(k_perturbation) ** 2
    D = a ** 4 + (4.0 * tau * a ** 2 + 8.0 * tau ** 2 * a * b) * k2
    om_p_sq = (a ** 2 + 2.0 * tau * k2 + np.sqrt(D + 0j)) / (2.0 * tau ** 2)
    om_m_sq = (a ** 2 + 2.0 * tau * k2 - np.sqrt(D + 0j)) / (2.0 * tau ** 2)
    return [np.sqrt(om_p_sq), -np.sqrt(om_p_sq),
            np.sqrt(om_m_sq), -np.sqrt(om_m_sq)], D


def most_unstable_mode(k_perturbation, kappa, sigma, p, tau):
    """Root Omega with the largest imaginary part (the growing mode)."""
    roots, _ = dispersion_roots(k_perturbation, kappa, sigma, p, tau)
    return roots[int(np.argmax([z.imag for z in roots]))]


def tau_band(kappa, sigma, p):
    """Threshold (T1): alpha < 0 <=> tau > tau_band.  inf if no such tau exists."""
    c = (p - 2.0) * kappa * sigma ** (p - 1.0)
    return -1.0 / c if c < 0 else np.inf


def k_critical(kappa, sigma, p, tau):
    """Lower edge k_c of the new unstable band; inf when the band is empty."""
    a, b, alpha = ab_alpha(kappa, sigma, p, tau)
    if alpha >= 0:
        return np.inf
    return np.sqrt(-a ** 3 / (4.0 * tau * a + 8.0 * tau ** 2 * b))


def gamma_inf(kappa, sigma, p, tau, k_big=1.0e6):
    """Large-k saturation value of the growth rate."""
    return most_unstable_mode(k_big, kappa, sigma, p, tau).imag


def report(k_perturbation, kappa, sigma, p, tau):
    """Print the stability diagnosis for one tau."""
    a, b, alpha = ab_alpha(kappa, sigma, p, tau)
    roots, D = dispersion_roots(k_perturbation, kappa, sigma, p, tau)
    Om = most_unstable_mode(k_perturbation, kappa, sigma, p, tau)
    kc = k_critical(kappa, sigma, p, tau)
    k2 = k_perturbation ** 2
    print(f"  tau={tau:<8g} a={a:+.4f}  b={b:+.4f}  alpha={alpha:+.4f}  D={D:+.4e}")
    print(f"    classical band (k^2 < 2ab = {2*a*b:+.4f}): "
          f"{'UNSTABLE' if k2 < 2*a*b else 'stable'}")
    print(f"    new band       (k   > k_c = {kc:.4f}):      "
          f"{'UNSTABLE' if k_perturbation > kc else 'stable'}")
    print(f"    Omega = {Om.real:+.5f} {Om.imag:+.5f}i    "
          f"Gamma_inf = {gamma_inf(kappa, sigma, p, tau):.5f}")
    return Om


# =====================================================================
# Initial data
# =====================================================================

def eigenmode_perturbation(x, k, kappa, sigma, p, tau, Om, amplitude):
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


# =====================================================================
if __name__ == "__main__":
    # ---------------- parameters ----------------
    L = 2 * np.pi
    N_points = 2 ** 10
    kappa = -1.0          # DEFOCUSING
    p = 3.0
    sigma = 1.0
    k = 0.0               # carrier wavenumber (k = 0 plane wave)
    k_perturbation = 8    # must be an integer multiple of 2 pi / L
    eps_p = 1.0e-3        # perturbation amplitude

    tau_list = [1.e-15, 0.9, 1.1, 1.5, 10.0]   # <1e-14 -> solver defaults to NLS
    Tfinal = 60.0
    dt = 5.e-3
    branch = '+'

    MAKE_GROWTH_RATE_FIGURE = True

    # ---------------- thresholds ----------------
    tb = tau_band(kappa, sigma, p)
    print("=" * 72)
    print(f"Defocusing NLSH:  p={p}, kappa={kappa}, sigma={sigma}, k_perturbation={k_perturbation}")
    print(f"  (T1) new band exists for tau > tau_band = {tb:.4f}")
    if np.isfinite(tb):
        taus = np.linspace(tb * (1 + 1e-9), 50.0, 200000)
        kcs = np.array([k_critical(kappa, sigma, p, t) for t in taus])
        inside = taus[kcs < k_perturbation]
        if inside.size:
            print(f"  (T2) mode k={k_perturbation} unstable for tau in "
                  f"[{inside.min():.4f}, {inside.max():.4f}]")
        else:
            print(f"  (T2) mode k={k_perturbation} is NEVER unstable "
                  f"(min k_c = {kcs.min():.3f}); increase k_perturbation")
    print("=" * 72)

    # ---------------- runs ----------------
    r_list, omega_list, amp0_list = [], [], []
    for tau in tau_list:
        print(f"Running simulation for tau={tau}")
        sol = NLSH1D(L=L, M=N_points, kappa=kappa, p=p, tau=tau)
        if tau < 1.e-10:
            # NLS limit: defocusing NLS is modulationally stable for every k
            Om = 0.0 + 0.0j
            print("    NLS limit: Omega^2 = k^4 - 2 kappa sigma^{p-1} k^2 > 0, stable")
        else:
            Om = report(k_perturbation, kappa, sigma, p, tau)
        omega_list.append(Om)

        q0i, q1i, om = plane_wave(sol.x, 0.0, k, sigma, kappa, p, tau, branch=branch)
        if Om.imag > 1.e-10:
            # seed the growing eigenmode -> clean exponential from t = 0
            de, dd = eigenmode_perturbation(sol.x, k_perturbation, kappa, sigma,
                                            p, tau, Om, eps_p)
            q0i = q0i + de
            q1i = q1i + dd
        else:
            # stable case: nothing to project onto, use a plain cosine
            q0i = q0i + eps_p * np.cos(k_perturbation * sol.x)

        amp0_list.append(np.max(np.abs(q0i)) - sigma)
        r = sol.run(q0i, q1i, Tfinal, dt, method="ars222", relaxation=True,
                    n_snapshots=600)
        print(f"    mass drift = {r.mass_drift:.3e}")
        r_list.append(r)

    x = sol.x

    # =================================================================
    # Figure 1: ||q0||_inf vs time, with the predicted growth envelope
    # =================================================================
    fs1, fs2 = 12, 15
    colors = ["#1ADF1A", '#FFC107', '#1E88E5', '#D81B60', '#FE6100']
    lw = 2

    fig, ax = plt.subplots(figsize=(6, 6.5))
    for idx, tau in enumerate(tau_list):
        r = r_list[idx]
        max_norm = [np.max(np.abs(q0)) for q0 in r.snapshots.q0]
        label_tau = r'$\tau$' + f'={tau}' if tau > 1.e-12 else 'NLS'
        ax.plot(r.snapshots.t, max_norm, label=label_tau, linewidth=lw,
                color=colors[idx])
        if omega_list[idx].imag > 1.e-10:
            envelope = amp0_list[idx] * np.exp(omega_list[idx].imag * r.snapshots.t)
            ax.plot(r.snapshots.t, sigma + envelope, linestyle='dashed',
                    color='black', linewidth=lw)

    ax.plot([], [], linestyle='dashed', color='black', linewidth=lw,
            label=r'$\Gamma_{\tau}(t)$')

    ax.set_ylim(0.9, 2.5)
    ax.set_xlabel('Time', fontsize=fs2)
    ax.set_ylabel(r'$\|q_0\|_\infty$', fontsize=fs2)
    ax.tick_params(axis='both', which='major', labelsize=fs1)

    n_entries = len(tau_list) + 1
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=min(n_entries, 3), fontsize=fs2,
              frameon=False, columnspacing=1.2, handlelength=1.8)
    fig.subplots_adjust(top=0.85)
    plt.savefig('defocusing_MI_threshold.pdf', bbox_inches='tight')


    # =================================================================
    # Figure 2: space-time contours
    # =================================================================
    fs1, fs2 = 12, 15
    x_scaled = sol.x / (2 * np.pi)
    X, T = np.meshgrid(x_scaled, r.snapshots.t)

    idxs = list(range(len(tau_list)))
    Qs = [np.abs(r_list[i].snapshots.q0) for i in idxs]
    cmap = 'RdBu'
    vmin, vmax = 0.5, 1.5
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(1, len(idxs), figsize=(6, 7), sharey=True,
                            gridspec_kw={'wspace': 0})
    for ax, i, Q in zip(axs, idxs, Qs):
        ax.pcolormesh(X, T, Q, norm=norm, cmap=cmap)
        ax.set_xlabel(r'$x/(2\pi)$', fontsize=fs2)
        ax.set_title(r'$\tau$=' + f'{tau_list[i]}' if tau_list[i] > 1.e-12 else 'NLS',
                     fontsize=fs2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', which='major', labelsize=fs1)
    axs[0].set_ylabel('Time', fontsize=fs2)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cbar_ax, label=r'$|q_0|$', extend='both')
    cbar.set_label(r'$|q_0|$', fontsize=fs2)
    cbar.ax.tick_params(labelsize=fs1)
    # plt.savefig('../paper/figures/defocusing_MI_contour.png',
    #             bbox_inches='tight', dpi=300)
    plt.savefig('defocusing_MI_contour.png',
                    bbox_inches='tight', dpi=300)