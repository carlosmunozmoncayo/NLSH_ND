#!/usr/bin/env python3
r"""
well_prepared_1d.py
===================
Does higher-order well-prepared initial data improve the rate at which NLSH
converges to NLS?  (1-D only.)

Setting
-------
NLSH (Section 2 of the paper, N = 1):

    i q0_t + (q1)_x + (V + kappa|q0|^{p-1}) q0 = 0
    i tau q1_t - (q0)_x + q1                   = 0

The slow manifold is  q1 = M_tau(q0)  with (paper, Sec. "Well-prepared
initial data")

    M_tau = d_x q0 - i tau <D M_tau, N_tau>,      N_tau = i(d_x M_tau + F[q0]),
    F[q0] = (V + kappa|q0|^{p-1}) q0.

Fixed-point iterates from M^0 = d_x q0:

    M^0 = q0_x
    M^1 = q0_x + tau d_x( q0_xx + F[q0] )
    M^2 = q0_x - i tau [ d_x N^1 + tau d_x( d_x^2 N^1 + DF[q0] N^1 ) ],
          N^1 = i( d_x M^1 + F[q0] ),
          DF[q0]h = V h + kappa (p+1)/2 |q0|^{p-1} h
                        + kappa (p-1)/2 |q0|^{p-3} q0^2 conj(h).

Initial data
------------
  groundstate  u_0 = sqrt(2 mu/kappa) sech(sqrt(mu) x); the NLS solution is
               known exactly, u = e^{i mu t} u_0  (defaults kappa=2, mu=1 =>
               u = sech(x) e^{i t}).  No reference error at all.
  gaussian     u_0 = exp(-x^2/(2 w^2)), w = --width (default 2); the NLS
               reference is computed with the same solver in its tau -> 0
               ('nls') mode at a finer dt, and the reference error is estimated
               by halving dt.  The leading error behaves like tau k^4 T (see
               the linear diagnostic), so a *narrow* gaussian (w = 1, spectrum
               out to k ~ 5) leaves the asymptotic regime entirely at these tau
               and the measured rate collapses; widen it or shrink tau.

Usage
-----
  python well_prepared_1d.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.append('../src/1d')
from nlsh1d import NLSH1D


# =====================================================================
# 1.  The slow-manifold maps  M^0, M^1, M^2
# =====================================================================

def dx(sol: NLSH1D, q: np.ndarray) -> np.ndarray:
    """Spectral d_x."""
    return np.fft.ifft(1j * sol.grid.k * np.fft.fft(q))


def dxx(sol: NLSH1D, q: np.ndarray) -> np.ndarray:
    """Spectral d_x^2 (uses the Nyquist-zeroed symbol, as the solver does)."""
    return np.fft.ifft(-(sol.grid.k ** 2) * np.fft.fft(q))


def F_nl(sol: NLSH1D, q0: np.ndarray) -> np.ndarray:
    """F[q0] = (V + kappa |q0|^{p-1}) q0."""
    f = sol.kappa * np.abs(q0) ** (sol.p - 1.0) * q0
    if sol.V is not None:
        f = f + sol.V * q0
    return f


def DF_nl(sol: NLSH1D, q0: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Frechet derivative of F at q0 in the (complex) direction h.

    DF[q0]h = V h + kappa (p+1)/2 |q0|^{p-1} h
                  + kappa (p-1)/2 |q0|^{p-3} q0^2 conj(h)

    R-linear but not C-linear: the conj(h) term is what makes M^2 differ from a
    naive complex-linear iteration.
    """
    a = np.abs(q0)
    p, kap = sol.p, sol.kappa
    with np.errstate(divide="ignore", invalid="ignore"):
        amp3 = np.where(a > 0.0, a ** (p - 3.0), 0.0)
    out = (kap * (p + 1.0) / 2.0 * a ** (p - 1.0) * h
           + kap * (p - 1.0) / 2.0 * amp3 * q0 ** 2 * np.conj(h))
    if sol.V is not None:
        out = out + sol.V * h
    return out


def dM1(sol: NLSH1D, q0: np.ndarray, h: np.ndarray) -> np.ndarray:
    """<D M^1[q0], h> = h_x + tau d_x( h_xx + DF[q0] h )."""
    return dx(sol, h) + sol.tau * dx(sol, dxx(sol, h) + DF_nl(sol, q0, h))


def slow_map(sol: NLSH1D, q0: np.ndarray, order: int) -> np.ndarray:
    """M^order(q0): the order-th fixed-point iterate of the slow-manifold map.

    order = 0 : q0_x                                     (the usual O(tau) data)
    order = 1 : q0_x + tau d_x(q0_xx + F)                (== well_prepared(2))
    order = 2 : q0_x - i tau <D M^1, N^1>
    """
    q0 = np.asarray(q0, dtype=complex)
    if order == 0:
        return dx(sol, q0)
    if order == 1:
        return dx(sol, q0) + sol.tau * dx(sol, dxx(sol, q0) + F_nl(sol, q0))
    if order == 2:
        m1 = slow_map(sol, q0, 1)
        n1 = 1j * (dx(sol, m1) + F_nl(sol, q0))
        return dx(sol, q0) - 1j * sol.tau * dM1(sol, q0, n1)
    raise ValueError("order must be 0, 1 or 2")


def _dir_deriv(f: Callable, q0: np.ndarray, h: np.ndarray,
               eps: float = 1e-3) -> np.ndarray:
    """<Df[q0], h> by a 4th-order central difference in a *real* step eps.

    Real eps is essential: these maps are only R-linear in the direction (the
    conj(h) in DF), so a complex step / holomorphic trick is not available.
    """
    return (-f(q0 + 2 * eps * h) + 8 * f(q0 + eps * h)
            - 8 * f(q0 - eps * h) + f(q0 - 2 * eps * h)) / (12.0 * eps)


def slow_map_residual(sol: NLSH1D, q0: np.ndarray, order: int,
                      eps: float = 1e-3) -> float:
    """L2 norm of the residual of the fixed-point problem

        R = M - q0_x + i tau <D M, N>,   N = i(d_x M + F[q0]).

    Should behave like O(tau^{order+1}).
    """
    M = slow_map(sol, q0, order)
    N = 1j * (dx(sol, M) + F_nl(sol, q0))
    DM = _dir_deriv(lambda q: slow_map(sol, q, order), q0, N, eps)
    return l2(sol, M - dx(sol, q0) + 1j * sol.tau * DM)


# =====================================================================
# 2.  small helpers
# =====================================================================

def l2(sol: NLSH1D, f: np.ndarray) -> float:
    """Discrete L^2(dx) norm."""
    return float(np.linalg.norm(f) * np.sqrt(sol.grid.dx))


def rates(taus, errs) -> np.ndarray:
    """Local convergence rates log(e_i/e_{i+1}) / log(t_i/t_{i+1})."""
    taus, errs = np.asarray(taus, float), np.asarray(errs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(errs[:-1] / errs[1:]) / np.log(taus[:-1] / taus[1:])
    return r


def fit_rate(taus, errs, last: int = 4) -> float:
    """Least-squares slope of log(err) vs log(tau) over the last few points."""
    taus, errs = np.asarray(taus, float)[-last:], np.asarray(errs, float)[-last:]
    good = np.isfinite(errs) & (errs > 0)
    if good.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(taus[good]), np.log(errs[good]), 1)[0])


def _fmt_row(label, vals, w=12, prec=3):
    return f"{label:>12s} " + " ".join(f"{v:{w}.{prec}e}" for v in vals)

# =====================================================================
# 3.  Initial data / reference solutions
# =====================================================================

class Problem:
    """Initial condition + a way to evaluate the NLS reference at time t."""

    def __init__(self, name, kappa, p, u0, exact=None, ref=None, L=60.0, M=512):
        self.name, self.kappa, self.p = name, kappa, p
        self.u0, self.exact, self.ref = u0, exact, ref
        self.L, self.M = L, M

    def u(self, sol, t_array) -> np.ndarray:
        """NLS solution on sol.x at the requested times, shape (nt, M)."""
        if self.exact is not None:
            return np.stack([self.exact(sol.x, t) for t in t_array])
        return self.ref(t_array)


def make_problem(kind: str, M: int, L: float, T: float, dt: float,
                 kappa: float, p: float, mu: float, width: float, method: str,
                 filter_kind: str, verbose=True) -> Problem:
    kind = kind.lower()
    if kind in ("groundstate", "soliton", "sech"):
        # -mu phi + phi_xx + kappa|phi|^{p-1}phi = 0 with p = 3:
        #   phi = sqrt(2 mu/kappa) sech(sqrt(mu) x),  u = e^{i mu t} phi
        if abs(p - 3.0) > 1e-12:
            raise ValueError("the closed-form ground state assumes p = 3")
        if kappa <= 0:
            raise ValueError("the ground state needs kappa > 0 (focusing)")
        amp, wid = np.sqrt(2.0 * mu / kappa), np.sqrt(mu)

        def u0(x):
            return amp / np.cosh(wid * x)

        def exact(x, t):
            return np.exp(1j * mu * t) * amp / np.cosh(wid * x)

        if verbose:
            print(f"  initial data : NLS ground state, kappa={kappa}, mu={mu}, "
                  f"u = e^(i mu t) sqrt(2mu/kappa) sech(sqrt(mu) x)")
            print(f"  tail at the boundary |u0(L/2)| = "
                  f"{amp/np.cosh(wid*L/2):.2e}  (periodic-truncation floor)")
        return Problem("groundstate", kappa, p, u0, exact=exact, L=L, M=M)

    if kind in ("gaussian", "gauss"):
        w = width

        def u0(x):
            return np.exp(-x ** 2 / (2.0 * w ** 2)).astype(complex)

        # numerical NLS reference on the same grid, finer dt, plus an error
        # estimate from halving dt.
        ref_cache: Dict[float, np.ndarray] = {}

        def _run_ref(dt_ref, ts):
            s = NLSH1D(M=M, L=L, x0=-L / 2, tau=0.0, kappa=kappa, p=p,
                       filter_kind=filter_kind)          # tau=0 -> 'nls' mode
            r = s.run(u0(s.x), None, T, dt_ref, method=method, relaxation=True,
                      n_snapshots=len(ts), land_on_snapshots=True)
            return r.snapshots.q0

        def ref(ts):
            key = tuple(np.round(ts, 12))
            if key not in ref_cache:
                dt_ref = dt / 4.0
                a = _run_ref(dt_ref, ts)
                b = _run_ref(dt_ref / 2.0, ts)
                err = np.max(np.abs(a - b)) * np.sqrt(L / M)
                if verbose:
                    print(f"  NLS reference: dt_ref = {dt_ref:.2e}, self-"
                          f"convergence |dt_ref vs dt_ref/2| = {err:.2e}")
                ref_cache[key] = b
            return ref_cache[key]

        if verbose:
            print(f"  initial data : gaussian u0 = exp(-x^2/(2*{w}^2)), "
                  f"kappa={kappa}, p={p}; NLS reference computed numerically")
            print(f"  spectral width ~ 1/w = {1/w:.2f}; the asymptotic regime "
                  f"needs tau k^4 T << 1, so broad data needs smaller tau")
        return Problem("gaussian", kappa, p, u0, ref=ref, L=L, M=M)

    raise ValueError(f"unknown initial condition '{kind}'")


# =====================================================================
# 4.  One NLSH run
# =====================================================================

def run_one(prob: Problem, tau: float, order: int, T: float, dt: float,
            M: int, L: float, method: str, filter_kind: str,
            n_snap: int, relaxation: bool = True):
    sol = NLSH1D(M=M, L=L, x0=-L / 2, tau=tau, kappa=prob.kappa, p=prob.p,
                 filter_kind=filter_kind)
    q0i = np.asarray(prob.u0(sol.x), dtype=complex)
    q1i = slow_map(sol, q0i, order)
    r = sol.run(q0i, q1i, T, dt, method=method, relaxation=relaxation,
                n_snapshots=n_snap, land_on_snapshots=True)
    return sol, r


# =====================================================================
# 5.  The convergence study
# =====================================================================

def convergence_study(prob: Problem, taus: List[float], orders: List[int],
                      T: float, dt: float, M: int, L: float, method: str,
                      filter_kind: str, n_snap: int) -> Dict:
    print()
    print("=" * 78)
    print(f"NLSH -> NLS with q1(0) = M^n(u_0),  T = {T}, dt = {dt}, "
          f"M = {M}, L = {L}")
    print(f"tableau = {method}, filter = {filter_kind}, "
          f"{n_snap} output times (L^inf_t norms)")
    print("=" * 78)

    res: Dict = {"taus": taus, "orders": orders, "T": T, "dt": dt,
                 "E": {n: [] for n in orders},      # max_t ||q0 - u||
                 "Thm": {n: [] for n in orders},    # + sqrt(tau)||q1 - u_x||
                 "Defect": {n: [] for n in orders},  # max_t ||q1 - M^2(q0)||
                 "Diff": {n: [] for n in orders[1:]},  # ||q0^n - q0^{n-1}||(T)
                 "profiles": {}}
    t0 = time.time()
    for tau in taus:
        q0T, sol_last = {}, None
        drift, toff = 0.0, 0.0
        for n in orders:
            sol, r = run_one(prob, tau, n, T, dt, M, L, method, filter_kind,
                             n_snap)
            s = r.snapshots
            u = prob.u(sol, s.t)
            ux = np.stack([dx(sol, uu) for uu in u])
            e = np.array([l2(sol, s.q0[i] - u[i]) for i in range(len(s))])
            e1 = np.array([l2(sol, s.q1[i] - ux[i]) for i in range(len(s))])
            dfc = np.array([l2(sol, s.q1[i] - slow_map(sol, s.q0[i], 2))
                            for i in range(len(s))])
            res["E"][n].append(float(e.max()))
            res["Thm"][n].append(float(e.max() + np.sqrt(tau) * e1.max()))
            res["Defect"][n].append(float(dfc.max()))
            q0T[n] = r.q0
            sol_last = sol
            drift = max(drift, r.mass_drift)
            toff = max(toff, float(np.max(np.abs(s.offsets))))
            if tau == taus[-1]:
                res["profiles"][n] = {"t": np.array(s.t), "err": e,
                                      "defect": dfc}
        for n in orders[1:]:
            res["Diff"][n].append(l2(sol_last, q0T[n] - q0T[n - 1]))
        print(f"  tau = {tau:9.3e} done   "
              + "  ".join(f"E(M^{n}) = {res['E'][n][-1]:.4e}" for n in orders)
              + f"   [mass drift {drift:.1e}, |t-t_out| {toff:.0e}]"
              + f" [{time.time()-t0:6.1f}s]")

    # ---------------- tables ----------------
    def table(title, key, labels):
        print()
        print(title)
        print(_fmt_row("tau", taus))
        for n in labels:
            print(_fmt_row(f"M^{n}", res[key][n]))
        for n in labels:
            r = rates(taus, res[key][n])
            print(f"  rate(M^{n}) = " + " ".join(f"{v:6.2f}" for v in r)
                  + f"    | fit = {fit_rate(taus, res[key][n]):5.2f}")

    table("max_t ||q0 - u||_L2   (the quantity in Thm 'convergence ND')",
          "E", orders)
    table("max_t ||q0-u|| + sqrt(tau) max_t ||q1 - u_x||", "Thm", orders)
    table("max_t ||q1 - M^2(q0)||_L2   (distance from the slow manifold)",
          "Defect", orders)
    if len(orders) > 1:
        print()
        print("||q0^(n) - q0^(n-1)||_L2 at t = T  (what the extra iterate buys)")
        print(_fmt_row("tau", taus))
        for n in orders[1:]:
            print(_fmt_row(f"M^{n}-M^{n-1}", res["Diff"][n]))
        for n in orders[1:]:
            r = rates(taus, res["Diff"][n])
            print(f"  rate(M^{n}-M^{n-1}) = " + " ".join(f"{v:6.2f}" for v in r)
                  + f"    | fit = {fit_rate(taus, res['Diff'][n]):5.2f}")

    # relative improvement in the actual error
    if len(orders) > 1:
        print()
        print("relative change in max_t||q0-u|| w.r.t. the M^0 run  "
              "(1 - E_n/E_0):")
        print(_fmt_row("tau", taus))
        for n in orders[1:]:
            imp = [1.0 - res["E"][n][i] / res["E"][0][i]
                   for i in range(len(taus))]
            print(f"{'M^'+str(n):>12s} " +
                  " ".join(f"{v:12.2%}" for v in imp))
    return res



# =====================================================================
# 6.  Plots
# =====================================================================

def make_plots(res: Dict, lin: Optional[Dict], prob: Problem,
               fname: str = "well_prepared_convergence") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    matplotlib.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    })

    taus = np.array(res["taus"])
    orders = res["orders"]
    cols = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    mrk = {0: "o", 1: "s", 2: "^"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # (a) error vs tau
    a = ax[0]
    for n in orders:
        a.loglog(taus, res["E"][n], mrk[n] + "-", color=cols[n],
                 label=rf"$q_1(0)=\mathcal{{M}}^{n}$", ms=5)
    c = res["E"][orders[0]][-1] / taus[-1]
    a.loglog(taus, c * taus, "k--", lw=1, label=r"$\mathcal{O}(\tau)$")
    a.set_xlabel(r"$\tau$")
    a.set_ylabel(r"$\max_t\|q_0-u\|_{L^2}$")
    a.set_title("Error relative to the NLS solution")
    a.legend()
    a.grid(True, which="both", alpha=0.25)

    # # (c) distance from the slow manifold
    c_ax = ax[1]
    for n in orders:
        c_ax.loglog(taus, res["Defect"][n], mrk[n] + "-", color=cols[n], ms=5,
                    label=rf"$\mathcal{{M}}^{n}$")
    for q, st in ((1, "--"), (2, ":"), (3, "-.")):
        if q - 1 < len(orders):
            k = orders[q - 1]
            cc = res["Defect"][k][-1] / taus[-1] ** q
            c_ax.loglog(taus, cc * taus ** q, "k" + st, lw=0.8,
                        label=rf"$\mathcal{{O}}(\tau^{q})$")
    c_ax.set_xlabel(r"$\tau$")
    c_ax.set_ylabel(r"$\max_t\|q_1-\mathcal{M}^2(q_0)\|_{L^2}$")
    c_ax.set_title("Distance from the slow manifold "+r"($\mathcal{M}^2$)")
    c_ax.legend(ncol=2)
    c_ax.grid(True, which="both", alpha=0.25)

    # fig.suptitle(f"Well-prepared initial data for NLSH (1D, {prob.name}, "
    #              f"T = {res['T']:g})")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fname}.{ext}", dpi=150, bbox_inches="tight")
    print(f"\n  wrote {fname}.pdf and {fname}.png")
    plt.close(fig)


# =====================================================================
# 10.  main
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Do higher-order well-prepared data help NLSH -> NLS? (1D)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ic", default="groundstate",
                    choices=["groundstate", "gaussian"])
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--M", type=int, default=512, help="grid points")
    ap.add_argument("--L", type=float, default=60.0, help="domain length")
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--p", type=float, default=3.0)
    ap.add_argument("--mu", type=float, default=1.0,
                    help="ground-state frequency (ic=groundstate)")
    ap.add_argument("--width", type=float, default=2.0,
                    help="gaussian width w in exp(-x^2/(2 w^2)) (ic=gaussian); "
                         "narrow data needs much smaller tau, see the notes")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3])
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2],
                    help="which M^n to use for q1(0)")
    ap.add_argument("--method", default="ars443",
                    choices=["imex_euler", "ars222", "cn_trap", "ars443"])
    ap.add_argument("--filter", dest="filter_kind", default="none",
                    choices=["none", "2/3", "exp", "hou-li"])
    ap.add_argument("--nsnap", type=int, default=41,
                    help="output times used for the L^inf_t norms")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="well_prepared_convergence")
    a = ap.parse_args(argv)

    orders = sorted(set(a.orders))
    if any(n not in (0, 1, 2) for n in orders):
        raise SystemExit("--orders must be a subset of 0 1 2")

    print()
    prob = make_problem(a.ic, a.M, a.L, a.T, a.dt, a.kappa, a.p, a.mu,
                        a.width, a.method, a.filter_kind)
    taus = sorted(a.taus, reverse=True)
    res = convergence_study(prob, taus, orders, a.T, a.dt, a.M, a.L,
                            a.method, a.filter_kind, a.nsnap)

    np.savez(a.out + ".npz",
             taus=np.array(taus), T=a.T, dt=a.dt, ic=a.ic,
             **{f"E{n}": np.array(res["E"][n]) for n in orders},
             **{f"Thm{n}": np.array(res["Thm"][n]) for n in orders},
             **{f"Defect{n}": np.array(res["Defect"][n]) for n in orders},
             **{f"Diff{n}": np.array(res["Diff"][n]) for n in orders[1:]})
    print(f"\n  wrote {a.out}.npz")

    if not a.no_plot:
        make_plots(res, None, prob, fname=a.out)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for n in orders:
        print(f"  q1(0) = M^{n}:  max_t||q0-u|| ~ C tau^{fit_rate(taus, res['E'][n]):.2f}"
              f"   (C ~ {res['E'][n][-1]/taus[-1]:.3g})")
    for n in orders[1:]:
        print(f"  M^{n} vs M^{n-1}:   solutions differ by "
              f"O(tau^{fit_rate(taus, res['Diff'][n]):.2f})")
    

if __name__ == "__main__":
    main()