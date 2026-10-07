r"""
    Convergence of NLSH solutions to the NLS solution as tau -> 0, with
    well-prepared initial data built from an NLS ground state.

  * Petviashvili's method is run ONCE per dimension with tau = 0 to get the
    NLS ground state ubar, i.e. the solution of  lap(ubar) - mu ubar + kappa ubar^p = 0.
  * The NLSH system is initialized with first-order well-prepared data

        q0(x,0) = ubar(x),      q1(x,0) = grad ubar(x)     ( = M_tau^0[q0] ),

    which is O(tau)-close to the NLSH slow manifold.
  * It is evolved to a fixed time T with the pseudo-spectral ImEx scheme
    (`nlshnd.NLSHND`, ARS443, relaxation on) for a decreasing sequence of tau.
  * The error is measured against the exact NLS standing wave

        u(x,t) = ubar(x) e^{i mu t},      grad u(x,t) = grad ubar(x) e^{i mu t},

    in the discrete L^infinity norm at t = T.
Usage
-----
    python verifying_thm.py --dry-run   # cost estimate
    python verifying_thm.py             # 2D and 3D
    python verifying_thm.py --dims 2d   # 2D only
    python verifying_thm.py --replot    # from cache

Results are cached in `./cache_ground_state_evolution/`, so re-running only
computes points that are missing and `--replot` restyles the figure.
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(thisdir, '2d/'))
sys.path.append(os.path.join(thisdir, '3d/'))
sys.path.append(os.path.join(thisdir, '../src/nd/'))

from petviashvili import run_and_save as run_and_save2d          # noqa: E402
from petviashvili3d import run_and_save as run_and_save3d        # noqa: E402
from nlshnd import NLSHND                                        # noqa: E402


# =====================================================================
# Parameters
# =====================================================================

MU, KAPPA, P = 1.0, 1.0, 3.0      # same as Section "Convergence of ground states"
LBOX = 20.0                       # periodic domain length in each direction
TAU_LIST = [10.0 ** (-i) for i in range(1, 8)]     # 1e-1 ... 1e-10

NITERS = 150                      # Petviashvili iterations (see note 1 above)
METHOD = "ars443"
WP_ORDER = 1                      # well-prepared order: 1 -> q1 = grad q0
                                  # (2 and 3 are available from
                                  #  NLSHND.well_prepared for the slow-manifold
                                  #  study of Section "well-prepared data")

# Per-dimension configuration.
#   T      : final time (3D must be short, see note 5)
#   m      : points per direction
#   dt_max : step used for the largest tau
#   dt_min : step used at tau = min(TAU_LIST); sets the time-discretization
#            floor, which is then measured and reported
CASES = {
    "2D": dict(N=2, m=128, T=5.0, dt_max=2.0e-3, dt_min=1.5e-4,
               marker="o", ls="-"),
    "3D": dict(N=3, m=48,  T=1.0, dt_max=1.0e-3, dt_min=1.0e-4,
               marker="s", ls="-"),
}

# Measured single-core cost per ImEx step, used only by --dry-run.
MS_PER_STEP = {"2D": 14.0, "3D": 57.0}

SAT_FRAC = 0.10        # error > SAT_FRAC * ||ubar||_inf  -> outside linear regime
FLOOR_FRAC = 3.0       # error < FLOOR_FRAC * floor       -> floor-contaminated
FLOOR_LINES = False     # draw the measured floors on the figure
OPEN_BAD_MARKERS = False

CACHE = os.path.join(thisdir, "cache_ground_state_evolution")
OUTPDF = os.path.join(thisdir, "error_standing_wave.pdf")

# Fontsizes (matching generate_focusing_MI_plots.py / the ground-state figure)
fs1 = 16
fs2 = 14

plt.rcParams.update({'font.size': fs1})

# =====================================================================
# Ground state and solver helpers
# =====================================================================

def nls_ground_state(case):
    """NLS ground state ubar on the periodic grid, from Petviashvili at tau=0.

    Returns a complex array of shape (m,)*N.  Only the field is used: the
    coordinate arrays returned by the Petviashvili modules are built with
    `linspace(-L/2, L/2, m)` while their FFT assumes spacing L/m, so the solver
    grid is rebuilt here from (L, m) instead (see note 3).
    """
    m, N = case["m"], case["N"]
    if N == 2:
        _, _, _, u, _ = run_and_save2d(tau=0.0, mu=MU, kappa=KAPPA,
                                       Lx=LBOX, Ly=LBOX, mx=m, my=m,
                                       niters=NITERS, save=False)
    else:
        _, _, _, _, u, _ = run_and_save3d(tau=0.0, mu=MU, kappa=KAPPA,
                                          Lx=LBOX, Ly=LBOX, Lz=LBOX,
                                          mx=m, my=m, mz=m,
                                          niters=NITERS, save=False)
    return np.asarray(u, dtype=complex)


def make_solver(case, tau):
    """NLSHND on the periodic box [-L/2, L/2)^N.  The de-aliasing filter MUST
    be off for ubar e^{i mu t} to be an exact semi-discrete solution (note 2)."""
    return NLSHND(M=case["m"], N=case["N"], L=LBOX, x0=-LBOX / 2.0,
                  tau=tau, kappa=KAPPA, p=P, filter_kind="none")


def steady_residual(sol, u):
    """||lap_h ubar - mu ubar + kappa ubar^p||_inf, with the solver's own
    Laplacian symbol.  This is the effective perturbation the evolution sees."""
    ur = u.real
    R = sol.lap(u).real - MU * ur + KAPPA * np.abs(ur) ** (P - 1.0) * ur
    return float(np.abs(R).max())


def nyquist_content(u):
    """|uhat| on a Nyquist hyperplane, relative to the DC coefficient.  Small
    means the Petviashvili and nlshnd Laplacians agree (note 3)."""
    uh = np.fft.fftn(u)
    nq = u.shape[0] // 2
    return float(np.abs(uh[nq]).max() / abs(uh.reshape(-1)[0]))


def timestep(tau, case):
    """dt ~ tau^(1/3), so the O(dt^3) floor stays a fixed fraction of the
    O(tau) error being plotted (note 4)."""
    tmin = min(TAU_LIST)
    dt = case["dt_min"] * (tau / tmin) ** (1.0 / 3.0)
    return float(np.clip(dt, case["dt_min"], case["dt_max"]))


# =====================================================================
# One (case, tau) measurement
# =====================================================================

def cache_path(label, case, tau):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(
        CACHE, f"{label}_m{case['m']}_T{case['T']:g}_wp{WP_ORDER}_"
               f"{METHOD}_tau{tau:.3e}.npz")


def run_one(label, case, tau, u, gu, controls, use_cache=True,
            require_cache=False):
    """Evolve NLSH from (ubar, grad ubar) to T and measure the errors.

    Returns dict with the two errors, the measured time-discretization floor,
    and the bookkeeping needed to flag unusable points.
    """
    path = cache_path(label, case, tau)
    if use_cache and os.path.exists(path):
        with np.load(path) as d:
            return {k: float(d[k]) for k in d.files}
    if require_cache:
        raise SystemExit(f"--replot: no cached result for {label}, tau={tau:g}\n"
                         f"  (expected {path})")

    T, dt = case["T"], timestep(tau, case)

    # --- NLS control at the SAME dt: measures the time-discretization floor,
    #     and doubles as the tau -> 0 reference trajectory.
    if dt not in controls:
        sol0 = make_solver(case, 0.0)          # tau = 0 -> solver's NLS mode
        r0 = sol0.run(u, None, T, dt, method=METHOD, relaxation=True,
                      store_every=0)
        ph0 = np.exp(1j * MU * r0.t)
        controls[dt] = dict(
            q0=r0.q0,
            gq0=sol0.grad(r0.q0),
            floor0=float(np.max(np.abs(r0.q0 - u * ph0))),
            floor1=float(np.max(np.abs(sol0.grad(r0.q0) - gu * ph0))),
            nsteps=r0.nsteps)
    ctl = controls[dt]

    sol = make_solver(case, tau)
    q1_0 = sol.well_prepared(u, order=WP_ORDER)     # order 1 -> grad ubar
    t0 = time.time()
    r = sol.run(u, q1_0, T, dt, method=METHOD, relaxation=True, store_every=0)
    wall = time.time() - t0

    ph = np.exp(1j * MU * r.t)
    out = dict(
        tau=tau, dt=dt, nsteps=float(r.nsteps), wall=wall,
        err_q0=float(np.max(np.abs(r.q0 - u * ph))),           # vs EXACT NLS
        err_q1=float(np.max(np.abs(r.q1 - gu * ph))),
        # same quantities with the time-discretization error cancelled, by
        # referencing the numerical NLS trajectory instead of the exact one
        dif_q0=float(np.max(np.abs(r.q0 - ctl["q0"]))),
        dif_q1=float(np.max(np.abs(r.q1 - ctl["gq0"]))),
        floor_q0=ctl["floor0"], floor_q1=ctl["floor1"],
        maxq0=float(np.max(np.abs(r.q0))))
    np.savez(path, **out)
    return out


def run_case(label, case, use_cache=True, require_cache=False):
    print("=" * 78)
    print(f"{label}: {case['N']}D, {case['m']}^{case['N']} grid, L = {LBOX}, "
          f"T = {case['T']}, mu = {MU}, kappa = {KAPPA}, p = {P}")
    print("=" * 78)

    u = nls_ground_state(case)
    sol0 = make_solver(case, 0.0)
    gu = sol0.grad(u)                       # grad ubar, the q1 reference
    res = steady_residual(sol0, u)
    nyq = nyquist_content(u.real)
    umax = float(np.abs(u).max())
    print(f"  ||ubar||_inf = {umax:.6f}")
    print(f"  Petviashvili steady-state residual ||R||_inf = {res:.2e}   "
          f"({NITERS} iterations)")
    print(f"  Nyquist content of ubar (relative)           = {nyq:.2e}")
    if res > 1e-9:
        print("  WARNING: residual is large; raise NITERS or refine the grid - "
              "it acts as an extra perturbation of this size.")
    if nyq > 1e-10:
        print("  WARNING: ubar is not spectrally resolved, so the Petviashvili "
              "and nlshnd Laplacians differ; refine the grid.")

    print(f"\n  {'tau':>9}  {'dt':>9} {'steps':>7}  {'err q0':>11} {'err q1':>11}"
          f"  {'floor q0':>10}  {'err/tau':>10}  flag")
    rows = []
    controls = {}
    for tau in TAU_LIST:
        d = run_one(label, case, tau, u, gu, controls, use_cache, require_cache)
        flag = ""
        if d["err_q0"] > SAT_FRAC * umax:
            flag = "saturated"
        elif d["err_q0"] < FLOOR_FRAC * d["floor_q0"]:
            flag = "floor-bound"
        d["ok"] = float(flag == "")
        rows.append(d)
        print(f"  {tau:9.1e}  {d['dt']:9.2e} {int(d['nsteps']):7d}  "
              f"{d['err_q0']:11.4e} {d['err_q1']:11.4e}  {d['floor_q0']:10.2e}  "
              f"{d['err_q0']/tau:10.3e}  {flag}")

    good = [r for r in rows if r["ok"]]
    if len(good) >= 2:
        lt = np.log10([r["tau"] for r in good])
        le = np.log10([r["err_q0"] for r in good])
        slope = np.polyfit(lt, le, 1)[0]
        print(f"\n  fitted order over the {len(good)} clean points: "
              f"error ~ tau^{slope:.3f}")
    else:
        print("\n  WARNING: fewer than two clean points - reduce T, refine the "
              "grid, or narrow the tau range.")
    return rows, umax


# =====================================================================
# Figure
# =====================================================================

def make_figure(results):
    plt.rcParams.update({'font.size': fs1})
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, (rows, _umax) in results.items():
        case = CASES[label]
        tau = np.array([r["tau"] for r in rows])
        e0 = np.array([r["err_q0"] for r in rows])
        e1 = np.array([r["err_q1"] for r in rows])
        ok = np.array([bool(r["ok"]) for r in rows])
        mk, ls = case["marker"], case["ls"]

        l0, = ax.loglog(tau, e0, mk + ls,
                        label=r'$\|q_0-u\|_\infty$ ' + label)
        l1, = ax.loglog(tau, e1, mk + ls,
                        label=r'$\|\mathbf{q}_1-\nabla u\|_\infty$ ' + label)
        if OPEN_BAD_MARKERS and (~ok).any():
            for line, e in ((l0, e0), (l1, e1)):
                ax.loglog(tau[~ok], e[~ok], mk, mfc='white',
                          mec=line.get_color(), ms=8, ls='none')
        if FLOOR_LINES:
            f0 = np.array([r["floor_q0"] for r in rows])
            ax.loglog(tau, f0, ':', lw=1, alpha=0.6, color=l0.get_color(),
                      label=r'$\Delta t$ floor ' + label)

    tt = np.array(TAU_LIST)
    slopes = [r["err_q0"] / r["tau"]
              for rows, _ in results.values() for r in rows if r["ok"]]
    anchor = max(slopes) if slopes else 1.0
    ax.loglog(tt, anchor * tt, 'k--', label=r'$\mathcal{O}(\tau)$')

    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$L^\infty$ error at $t=T$')
    fig.suptitle('Convergence of well-prepared NLSH solutions as '
                 + r' $\tau \to 0$')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(OUTPDF)), exist_ok=True)
    fig.savefig(OUTPDF, bbox_inches='tight')
    print(f"\nwrote {OUTPDF}")
    return fig


def dry_run():
    print("Planned work (dt ~ tau^(1/3); the NLS control at each distinct dt "
          "is shared across tau):\n")
    total = 0
    for label, case in CASES.items():
        seen, steps = set(), 0
        print(f"  {label}  {case['m']}^{case['N']}, T = {case['T']}")
        for tau in TAU_LIST:
            dt = timestep(tau, case)
            n = int(np.ceil(case["T"] / dt))
            steps += n
            if dt not in seen:
                seen.add(dt)
                steps += n            # the NLS control at this dt
            print(f"    tau = {tau:8.1e}   dt = {dt:9.2e}   {n:8d} steps")
        ms = MS_PER_STEP.get(label, 20.0)
        print(f"    -> {steps} steps total ({len(seen)} distinct dt), "
              f"~{steps*ms/1000/60:.0f} min at {ms:g} ms/step\n")
        total += steps
    print(f"  grand total {total} steps.  The per-step costs above were measured "
          f"on one core\n  (2D 128^2 ~14 ms, 3D 48^3 ~57 ms); scale them for "
          f"your machine.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--dims", nargs="+", default=list(CASES),
                    type=str.upper, choices=list(CASES),
                    metavar="{2d,3d}")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the step counts and exit")
    ap.add_argument("--replot", action="store_true",
                    help="use only cached results (fails on a missing point)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tau-min", type=float, default=None,
                    help="drop tau below this (e.g. 1e-8 for a quicker run)")
    ap.add_argument("--T2d", type=float, default=None)
    ap.add_argument("--T3d", type=float, default=None)
    ap.add_argument("--m2d", type=int, default=None)
    ap.add_argument("--m3d", type=int, default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    global TAU_LIST
    if a.tau_min is not None:
        TAU_LIST = [t for t in TAU_LIST if t >= a.tau_min * (1 - 1e-12)]
    if a.T2d is not None:
        CASES["2D"]["T"] = a.T2d
    if a.T3d is not None:
        CASES["3D"]["T"] = a.T3d
    if a.m2d is not None:
        CASES["2D"]["m"] = a.m2d
    if a.m3d is not None:
        CASES["3D"]["m"] = a.m3d

    if a.dry_run:
        dry_run()
        return

    if not a.show:
        matplotlib.use("Agg")

    results = {}
    for label in a.dims:
        rows, umax = run_case(label, CASES[label],
                              use_cache=not a.no_cache,
                              require_cache=a.replot)
        results[label] = (rows, umax)

    make_figure(results)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()