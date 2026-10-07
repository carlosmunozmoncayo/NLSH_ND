r"""
nlsh1d.py
=========
Fourier-pseudospectral / IMEX-ESDIRK solver for the 1-D first-order hyperbolic
approximation (NLSH) of the nonlinear Schroedinger equation.

Model (1-D, N = 1)
------------------
    i q0_t + (q1)_x + (V(x) + kappa |q0|^{p-1}) q0 = 0
    i tau q1_t - (q0)_x + q1                       = 0

solved for (q0, q1) on a periodic interval.  In evolution form

    q0_t = i (q1)_x + i kappa |q0|^{p-1} q0 + i V q0
    q1_t = -(i/tau) (q0)_x + (i/tau) q1

which we split as  U_t = L(U) + N(U)  with

    L(U) = ( i d_x q1 ,  -(i/tau) d_x q0 + (i/tau) q1 )   <- stiff, LINEAR  -> implicit
    N(U) = ( i (kappa |q0|^{p-1} + V) q0 , 0 )            <- non-stiff      -> explicit

Space
-----
Fourier pseudospectral, d_x -> i k. 
L is *diagonal* in Fourier space: for each wavenumber k

    d/dt [q0h, q1h]^T = M(k) [q0h, q1h]^T ,   M(k) = [[0, -k], [k/tau, i/tau]]

so every implicit stage is an exact 2x2 solve per mode.

Time
----
IMEX Runge-Kutta, implicit tableau (A, b) is an ESDIRK (a_11 = 0,
a_ii = gamma for i >= 2), explicit tableau (Ah, bh) is an ERK, both sharing
the same abscissae c = ch.  All bundled tableaux are *globally stiffly
accurate* (GSA):

    b_i = a_{s,i},   bh_i = ah_{s,i},   c_s = ch_s = 1   =>   U^{n+1} = Y_s

so the last stage *is* the update (the code exploits this and skips the final
L/N evaluations).  GSA is what makes the scheme asymptotic-preserving as
tau -> 0: the numerical solution is projected onto the same "equilibrium
manifold" q1 = d_x q0 + O(tau) that the exact solution relaxes to.

Invariant
---------
    Mtau(U) = int |q0|^2 + tau |q1|^2 dx

is conserved by the PDE: L is skew-adjoint w.r.t. the weighted inner product
    <U,V>_A = Re int ( conj(q0) v0 + tau conj(q1) v1 ) dx
and N is orthogonal to U in the same product.  Both properties survive the
pseudospectral semi-discretisation exactly (i k is a purely imaginary symbol;
the nonlinearity is diagonal in physical space).  Only the RK time stepping
breaks it.  Setting relaxation=True restores it exactly via
relaxation: with the increment  d = (U* - U^n)/dt  produced by the IMEX step,

    U^{n+1} = U^n + gamma dt d ,   t^{n+1} = t^n + gamma dt ,
    gamma   = -2 <U^n, d>_A / ( dt <d, d>_A )   ( = 1 + O(dt^{order-1}) ),

which makes Mtau(U^{n+1}) = Mtau(U^n) to machine precision while retaining
the full order of the underlying scheme.

NLS fallback
------------
If tau <= tau_nls (default 1e-14) the solver switches to `mode = 'nls'` and
integrates the limit equation itself,

    i q0_t + q0_xx + (V + kappa|q0|^{p-1}) q0 = 0 ,

on one component: the state is (1, M) instead of (2, M), `run` ignores q1, and
`unpack` returns q1 = None.  The stiff part is L = -i k^2 in Fourier space, so
the implicit solve is a division by (1 + i a k^2) - which is exactly what the
2x2 NLSH solve above degenerates to as tau -> 0.  The Nyquist-zeroed symbol
`grid.k` is used for the Laplacian too, so NLS mode is the exact tau -> 0 limit
of the NLSH mode mode by mode.

Recorded diagnostics
--------------------
Alongside `mass_tau` (the relaxed invariant), `hist` and `snapshots` carry

    mass_q0 : int |q0|^2 dx                       (NLS mass, drifts O(tau))
    energy  : E_NLSH = E_NLS - || d_x q0 - q1 ||_{L^2}^2 ,
              E_NLS = int |d_x q0|^2 - V|q0|^2 - (2k/(p+1))|q0|^{p+1} dx.
              Equal to E_NLS in NLS mode.  Method: `energy(U)`.
    virial  : Jtilde_tau = int x^2 (|q0|^2 + tau|q1|^2) dx
                           + 4 tau Re int conj(q0) (x q1) dx .
              Equal to int x^2|q0|^2 in NLS mode.  Method: `virial(U)`.
              x^2 is not periodic - build the solver with x0 = -L/2 and keep
              the solution away from the boundary for this to mean anything.
    max_q0  : ||q0||_inf

These cost a few FFTs per recorded step; raise `store_every` if that matters.

Filtering
---------
The de-aliasing filter is applied *symmetrically*, N(U) = S(f(S q0)), with S a
real symmetric Fourier multiplier.  Because S is then self-adjoint,
<U, N(U)>_A = Re int conj(S q0) i(...)|S q0|^{p-1} S q0 = 0 still holds, i.e.
filtering does not destroy the conservation structure.

Both are implemented/verified in `plane_wave` and `test_plane_wave_residual`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "IMEXTableau", "TABLEAUX", "get_tableau", "check_tableau",
    "stability_function", "NLSH1D", "Result", "plane_wave",
]


# =====================================================================
# 1.  IMEX Butcher tableaux
# =====================================================================

@dataclass
class IMEXTableau:
    """An IMEX-RK pair.  A/b = implicit (ESDIRK), Ah/bh = explicit (ERK)."""
    name: str
    A: np.ndarray
    b: np.ndarray
    Ah: np.ndarray
    bh: np.ndarray
    order: int
    stability: str = ""          # 'L-stable', 'A-stable', ...
    notes: str = ""

    def __post_init__(self):
        self.A = np.array(self.A, dtype=float)
        self.b = np.array(self.b, dtype=float)
        self.Ah = np.array(self.Ah, dtype=float)
        self.bh = np.array(self.bh, dtype=float)
        self.s = self.b.size
        self.c = self.A.sum(axis=1)
        self.ch = self.Ah.sum(axis=1)

    @property
    def gsa(self) -> bool:
        """Globally stiffly accurate: U^{n+1} == Y_s for both parts."""
        return bool(np.allclose(self.b, self.A[-1]) and
                    np.allclose(self.bh, self.Ah[-1]) and
                    np.isclose(self.c[-1], 1.0) and np.isclose(self.ch[-1], 1.0))

    @property
    def gamma(self) -> float:
        """The (constant) nonzero diagonal entry of the implicit tableau."""
        d = np.diag(self.A)
        nz = d[np.abs(d) > 1e-14]
        return float(nz[0]) if nz.size else 0.0

    def __repr__(self):
        return (f"IMEXTableau({self.name}, s={self.s}, order={self.order}, "
                f"GSA={self.gsa}, {self.stability})")


def _tab_imex_euler() -> IMEXTableau:
    """Backward Euler / forward Euler.  ARS(1,1,1)."""
    A = [[0., 0.],
         [0., 1.]]
    b = [0., 1.]
    Ah = [[0., 0.],
          [1., 0.]]
    bh = [1., 0.]
    return IMEXTableau("imex_euler", A, b, Ah, bh, order=1,
                       stability="L-stable",
                       notes="Cheapest option; useful as a robust fallback.")


def _tab_ars222() -> IMEXTableau:
    """Ascher-Ruuth-Spiteri (2,2,2).  2nd order, L-stable, GSA."""
    g = (2.0 - np.sqrt(2.0)) / 2.0
    d = 1.0 - 1.0 / (2.0 * g)              # = -sqrt(2)/2
    A = [[0., 0.,    0.],
         [0., g,     0.],
         [0., 1 - g, g]]
    b = [0., 1 - g, g]
    Ah = [[0., 0.,    0.],
          [g,  0.,    0.],
          [d,  1 - d, 0.]]
    bh = [d, 1 - d, 0.]
    return IMEXTableau("ars222", A, b, Ah, bh, order=2,
                       stability="L-stable",
                       notes="Workhorse 2nd-order IMEX; strong damping of the "
                             "spurious O(1/tau) mode.")


def _tab_cn_trap() -> IMEXTableau:
    """Crank-Nicolson + explicit trapezoidal.  2nd order, A-stable, GSA.

    R(inf) = -1, so the fast relaxation mode is *not* damped, but the implicit
    part conserves the quadratic invariant exactly (CN is symplectic-like for
    skew-adjoint L).  Best choice when you want relaxation gammas ~ 1 + 1e-12.
    """
    A = [[0.0, 0.0, 0.0],
         [0.5, 0.5, 0.0],
         [0.5, 0.0, 0.5]]
    b = [0.5, 0.0, 0.5]
    Ah = [[0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.5, 0.5, 0.0]]
    bh = [0.5, 0.5, 0.0]
    return IMEXTableau("cn_trap", A, b, Ah, bh, order=2,
                       stability="A-stable (R(inf) = -1)",
                       notes="Non-dissipative implicit part: quadratic "
                             "invariant exactly conserved for the linear part.")


def _tab_ars443() -> IMEXTableau:
    """Ascher-Ruuth-Spiteri (4,4,3).  3rd order, L-stable, GSA."""
    A = [[0.,  0.,    0.,   0.,  0.],
         [0.,  1/2,   0.,   0.,  0.],
         [0.,  1/6,   1/2,  0.,  0.],
         [0., -1/2,   1/2,  1/2, 0.],
         [0.,  3/2,  -3/2,  1/2, 1/2]]
    b = [0., 3/2, -3/2, 1/2, 1/2]
    Ah = [[0.,     0.,    0.,   0.,   0.],
          [1/2,    0.,    0.,   0.,   0.],
          [11/18,  1/18,  0.,   0.,   0.],
          [5/6,   -5/6,   1/2,  0.,   0.],
          [1/4,    7/4,   3/4, -7/4,  0.]]
    bh = [1/4, 7/4, 3/4, -7/4, 0.]
    return IMEXTableau("ars443", A, b, Ah, bh, order=3,
                       stability="L-stable",
                       notes="Default: 3rd order, 4 implicit solves per step.")


TABLEAUX: Dict[str, Callable[[], IMEXTableau]] = {
    "imex_euler": _tab_imex_euler,
    "ars222":     _tab_ars222,
    "cn_trap":    _tab_cn_trap,
    "ars443":     _tab_ars443,
}


def get_tableau(name_or_tab) -> IMEXTableau:
    if isinstance(name_or_tab, IMEXTableau):
        return name_or_tab
    try:
        return TABLEAUX[name_or_tab]()
    except KeyError:
        raise KeyError(f"unknown tableau '{name_or_tab}'. "
                       f"Available: {sorted(TABLEAUX)}") from None


def register_tableau(tab: IMEXTableau) -> None:
    """Add your own pair; `check_tableau` will vet it."""
    TABLEAUX[tab.name] = lambda t=tab: t


# ---------------------------------------------------------------------
# tableau verification (structure + IMEX order conditions up to order 3)
# ---------------------------------------------------------------------

def stability_function(tab: IMEXTableau, z: complex) -> complex:
    """R(z) of the *implicit* tableau."""
    s = tab.s
    return 1.0 + z * tab.b @ np.linalg.solve(np.eye(s) - z * tab.A, np.ones(s))


def check_tableau(tab, tol: float = 1e-12, verbose: bool = True) -> Dict[str, float]:
    """Verify structure, GSA and coupled IMEX order conditions.

    Returns a dict {condition: residual}.  Everything should be ~0 / True.
    """
    tab = get_tableau(tab)
    out: Dict[str, float] = {}

    struct = {
        "Ah strictly lower triangular (explicit)": np.abs(np.triu(tab.Ah)).max(),
        "A lower triangular":                      np.abs(np.triu(tab.A, 1)).max(),
        "c == ch":                                 np.abs(tab.c - tab.ch).max(),
        "b  == A[-1]  (stiffly acc., implicit)":   np.abs(tab.b - tab.A[-1]).max(),
        "bh == Ah[-1] (stiffly acc., explicit)":   np.abs(tab.bh - tab.Ah[-1]).max(),
        "c_s == 1":                                abs(tab.c[-1] - 1.0),
    }
    d = np.diag(tab.A)
    nz = d[np.abs(d) > tol]
    struct["SDIRK (constant diagonal)"] = float(np.abs(nz - nz[0]).max()) if nz.size else 0.0
    out.update(struct)

    B = {"b": tab.b, "bh": tab.bh}
    C = {"c": tab.c, "ch": tab.ch}
    AA = {"A": tab.A, "Ah": tab.Ah}

    cond: Dict[str, float] = {}
    for nb, bb in B.items():
        cond[f"sum({nb}) = 1"] = bb.sum() - 1.0
    if tab.order >= 2:
        for nb, bb in B.items():
            for nc, cc in C.items():
                cond[f"{nb}.{nc} = 1/2"] = bb @ cc - 0.5
    if tab.order >= 3:
        for nb, bb in B.items():
            for nc, cc in C.items():
                cond[f"{nb}.{nc}^2 = 1/3"] = bb @ (cc * cc) - 1.0 / 3.0
                for na, aa in AA.items():
                    cond[f"{nb}.{na}.{nc} = 1/6"] = bb @ (aa @ cc) - 1.0 / 6.0
    out.update(cond)

    rinf = stability_function(tab, -1e14)
    out["|R(inf)|"] = abs(rinf)

    if verbose:
        print(f"--- {tab.name}:  s = {tab.s}, order = {tab.order}, "
              f"gamma = {tab.gamma:.10f}, GSA = {tab.gsa}")
        worst = max(abs(v) for k, v in cond.items()) if cond else 0.0
        print(f"    structure: " + ", ".join(
            f"{k.split('(')[0].strip()}={'OK' if abs(v) < tol else 'FAIL(%.1e)' % v}"
            for k, v in struct.items()))
        print(f"    order conditions up to order {tab.order}: "
              f"{len(cond)} checked, max residual = {worst:.2e} "
              f"({'PASS' if worst < 1e-12 else 'FAIL'})")
        print(f"    |R(inf)| = {abs(rinf):.3e}  "
              f"({'L-stable' if abs(rinf) < 1e-8 else 'not L-stable'})")
    return out


# =====================================================================
# 2.  Fourier grid + filters
# =====================================================================

class Fourier1D:
    """Periodic grid on [x0, x0+L) with M points and the FFT wavenumbers."""

    def __init__(self, M: int, L: float, x0: float = 0.0):
        self.M, self.L, self.x0 = int(M), float(L), float(x0)
        self.dx = self.L / self.M
        self.x = self.x0 + self.dx * np.arange(self.M)
        self.k_raw = 2.0 * np.pi * np.fft.fftfreq(self.M, d=self.dx)
        self.k = self.k_raw.copy()
        if self.M % 2 == 0:                 # kill Nyquist in the odd-derivative symbol
            self.k[self.M // 2] = 0.0
        self.kmax = np.pi / self.dx


def make_filter(grid: Fourier1D, kind: str = "2/3",
                alpha: float = 36.0, order: int = 8) -> np.ndarray:
    """Real, symmetric Fourier multiplier S(k), applied symmetrically in N(U).

    kind = 'none'   : S = 1
           '2/3'    : sharp cut at (2/3) k_max  (exact de-aliasing of a cubic)
           'exp'    : exp(-alpha (|k|/kmax)^(2*order))     (smooth, spectral)
           'hou-li' : exp(-36 (|k|/kmax)^36)               (very flat, mild)
    """
    kk = np.abs(grid.k_raw) / grid.kmax
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return np.ones_like(kk)
    if kind in ("2/3", "23", "twothirds", "dealias"):
        return (kk <= 2.0 / 3.0).astype(float)
    if kind == "exp":
        return np.exp(-alpha * kk ** (2 * order))
    if kind in ("hou-li", "houli"):
        return np.exp(-36.0 * kk ** 36)
    raise ValueError(f"unknown filter '{kind}'")


# =====================================================================
# 3.  The solver
# =====================================================================

@dataclass
class Snapshots:
    """Full solution stored at ~equally spaced output times.

    All arrays have leading dimension n = number of snapshots (the first entry
    is always the initial condition at t = 0, the last one is the end of the
    run).  q0/q1 have shape (n, M) and live in *physical* space.

        t         (n,)    actual time of the snapshot
        dt        (n,)    time actually advanced by the step that produced it
                          ( = gamma * dt_requested; t[0] entry is 0)
        gamma     (n,)    relaxation parameter of that step
        mass_tau  (n,)    the quadratic invariant of the solver
        step      (n,)    step index at which it was recorded
        diag      dict    solver-specific extras, also reachable as attributes
                          (e.g. .mass_q0 = int |q0|^2 dx, .max_q0, .energy)
        t_target  (n,)    the requested (exactly equispaced) output time

    Unless land_on_snapshots=True the recorded t are the first step-ends at or
    past t_target, so they are equispaced only up to one step; t_target - t is
    reported by `.offsets`.  Two cases give fewer than n_snapshots entries: a
    step longer than the target spacing (possible only with an adaptive dt)
    covers several targets and is stored once, and a run halted by stop() is
    truncated - with the state at the moment of stopping appended, its
    t_target being NaN.
    """
    t: np.ndarray
    q0: np.ndarray
    q1: Optional[np.ndarray]            # None when there is no q1 (NLS mode)
    dt: np.ndarray
    gamma: np.ndarray
    mass_tau: np.ndarray
    step: np.ndarray
    t_target: np.ndarray
    diag: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self):
        return self.t.size

    def __getattr__(self, name):
        """Expose the solver-specific diagnostics as attributes (.mass_q0 ...)."""
        d = self.__dict__.get("diag")
        if d is not None and name in d:
            return d[name]
        raise AttributeError(name)

    def __getitem__(self, i):
        base = {k: getattr(self, k)[i] for k in
                ("t", "q0", "dt", "gamma", "mass_tau", "step", "t_target")}
        base["q1"] = None if self.q1 is None else self.q1[i]
        base.update({k: v[i] for k, v in self.diag.items()})
        return base

    @property
    def offsets(self) -> np.ndarray:
        """t - t_target: how far each snapshot sits from the exact output time."""
        return self.t - self.t_target

    @property
    def abs_q0(self) -> np.ndarray:
        """|q0| as an (n, M) array - handy for space-time plots."""
        return np.abs(self.q0)

    def __repr__(self):
        return (f"Snapshots(n={len(self)}, t in [{self.t[0]:.4g}, {self.t[-1]:.4g}], "
                f"max|t - t_target| = {np.max(np.abs(self.offsets)):.2e})")


@dataclass
class Result:
    """Output of NLSH1D.run."""
    t: float
    q0: np.ndarray
    q1: np.ndarray
    U: np.ndarray                       # Fourier coefficients, shape (2, M)
    hist: Dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: Optional[Snapshots] = None
    nsteps: int = 0
    tableau: str = ""

    def __repr__(self):
        ns = f", snapshots={len(self.snapshots)}" if self.snapshots else ""
        return (f"Result(t={self.t:.6g}, nsteps={self.nsteps}, "
                f"tableau={self.tableau}{ns}, "
                f"mass drift={self.mass_drift:.3e})")

    @property
    def mass_drift(self) -> float:
        m = self.hist.get("mass_tau")
        return float(np.max(np.abs(m - m[0])) / abs(m[0])) if m is not None else np.nan


# =====================================================================
# 3a. Time-stepping core (spatial-discretisation agnostic)
# =====================================================================

class IMEXIntegrator:
    """IMEX-RK time stepping with relaxation and snapshots.

    A subclass supplies the space discretisation through the hooks below; the
    stepping, the relaxation and the output machinery are shared.  Both the
    Fourier solver (NLSH1D) and the mapped-grid solver (nlsh_mapped.py) use it.

        Lop(U)                 stiff linear part, applied to the state
        Nop(U)                 non-stiff (explicit) part
        solve_implicit(r, a)   solve (I - a Lop) Y = r
        inner(U, V)            the real inner product the invariant comes from
        pack(q0, q1)/unpack(U) state <-> physical fields
        diagnostics(U)         extra scalars recorded per step/snapshot
    """

    def Lop(self, U):                       raise NotImplementedError
    def Nop(self, U):                       raise NotImplementedError
    def solve_implicit(self, rhs, a):       raise NotImplementedError
    def inner(self, U, V) -> float:         raise NotImplementedError
    def pack(self, q0, q1=None):            raise NotImplementedError
    def unpack(self, U):                    raise NotImplementedError

    def diagnostics(self, U) -> Dict[str, float]:
        """Extra scalars stored in hist/snapshots (keys must not change)."""
        return {}

    def mass_tau(self, U) -> float:
        """The quadratic invariant, <U,U>."""
        return self.inner(U, U)

    # -- one IMEX step -------------------------------------------------
    def step(self, U: np.ndarray, dt: float, tab: IMEXTableau,
             relaxation: bool = False) -> Tuple[np.ndarray, float]:
        """One IMEX-RK step.

        Returns (d, gamma) with d = (U* - U^n)/dt the increment of the
        unrelaxed scheme, so that  U^{n+1} = U^n + gamma*dt*d.
        gamma == 1.0 when relaxation is off.
        """
        s = tab.s
        Ls: list = [None] * s
        Ns: list = [None] * s
        Unew = None

        for i in range(s):
            rhs = U.copy()
            for j in range(i):
                aij, ahij = tab.A[i, j], tab.Ah[i, j]
                if aij != 0.0:
                    rhs += (dt * aij) * Ls[j]
                if ahij != 0.0:
                    rhs += (dt * ahij) * Ns[j]
            Y = self.solve_implicit(rhs, dt * tab.A[i, i])
            if tab.gsa and i == s - 1:
                Unew = Y                      # GSA: last stage IS the update
                break
            Ls[i] = self.Lop(Y)
            Ns[i] = self.Nop(Y)
        else:                                  # non-GSA fallback
            Unew = U.copy()
            for i in range(s):
                if tab.b[i] != 0.0:
                    Unew += (dt * tab.b[i]) * Ls[i]
                if tab.bh[i] != 0.0:
                    Unew += (dt * tab.bh[i]) * Ns[i]

        d = (Unew - U) / dt
        gamma = 1.0
        if relaxation:
            # gamma solves  eta(U + gamma dt d) = eta(U):
            #   gamma = -2 <U,d> / (dt <d,d>) = 1 + O(dt^{order-1}).
            # <U,d> is O(dt) and computed as a sum of O(1) terms, so its
            # relative accuracy is ~ eps_mach/dt: very small dt is hopeless.
            dd = self.inner(d, d)
            if dd > 0.0:
                gamma = -2.0 * self.inner(U, d) / (dt * dd)
        return d, gamma

    # -- time loop -----------------------------------------------------
    def run(self, q0_0, q1_0=None, T: float = 1.0, dt=1e-2,
            method="ars443",
            relaxation: bool = True, relaxation_variant: str = "relaxation",
            land_on_T: bool = True, gamma_tol: float = 0.2,
            callback: Optional[Callable] = None, store_every: int = 1,
            n_snapshots: Optional[int] = None, land_on_snapshots: bool = False,
            stop: Optional[Callable] = None, max_steps: int = 10_000_000,
            verbose: bool = False) -> Result:
        """Integrate from t=0 to t=T.

        dt                  : a float, or a callable dt(t, U, solver) -> float
                              evaluated at the start of every step.  The
                              callable form is what blow-up needs: the explicit
                              nonlinearity is stable only for
                              dt <~ C/(kappa ||q0||_inf^(p-1)), which shrinks as
                              the solution grows.
        relaxation          : enforce exact conservation of int |q0|^2+tau|q1|^2
        relaxation_variant  : 'relaxation' -> t <- t + gamma*dt (keeps order p)
                              'idt'        -> t <- t + dt       (loses one order)
        land_on_T           : with relaxation, solve gamma(dt)*dt = T-t by fixed
                              point on the final step so the run ends exactly at
                              T without giving up conservation.
        n_snapshots         : store the full solution (q0, q1) plus t, dt, gamma
                              and both masses at this many ~equally spaced times,
                              *including* t=0 and the end of the run (so
                              n_snapshots >= 2).  None/0 -> final state only.
                              Memory: n_snapshots * M * 32 bytes.
        land_on_snapshots   : False (default) - relaxation is never interfered
                              with: each snapshot is taken at the first step end
                              at or past its target time, so output times are
                              equispaced only up to one step (see
                              Snapshots.offsets).  True - additionally hit every
                              output time exactly, by the same conservative
                              fixed point used for T; this keeps exact
                              conservation but reshuffles the step sizes.
        gamma_tol           : warn if |gamma-1| exceeds this (see module notes).
        callback(t, U, self): called after every accepted step.
        stop(t, U, self)    : optional predicate; the run ends early (cleanly,
                              with everything recorded so far) the first time it
                              returns True.  Blow-up wants a threshold on
                              ||q0||_inf rather than a final time.
        max_steps           : hard cap, as a guard against a collapsing dt.
        """
        tab = get_tableau(method)
        if not tab.gsa:
            warnings.warn(f"tableau '{tab.name}' is not globally stiffly accurate")
        variant = relaxation_variant.lower()
        if variant not in ("relaxation", "idt"):
            raise ValueError("relaxation_variant must be 'relaxation' or 'idt'")
        if relaxation and tab.order < 2:
            warnings.warn(
                f"relaxation with the first-order '{tab.name}': gamma = "
                "1 + O(dt^(p-1)) does NOT tend to 1 when p = 1, so the step is "
                "rescaled by an O(1) factor.  Use a >= 2nd-order tableau.")

        U = self.pack(q0_0, q1_0)
        t, n = 0.0, 0
        eps = 1e-12 * max(1.0, abs(T))
        hist = {"t": [0.0], "dt": [0.0], "gamma": [1.0],
                "mass_tau": [self.mass_tau(U)]}
        for key, val in self.diagnostics(U).items():
            hist[key] = [val]
        gmax, n_clamped = 0.0, 0

        # ---- snapshot bookkeeping -----------------------------------
        snap_t = None
        if n_snapshots:
            if n_snapshots < 2:
                raise ValueError("n_snapshots counts t=0 and t=T, so it must be >= 2")
            snap_t = np.linspace(0.0, T, int(n_snapshots))
            d_snap = snap_t[1] - snap_t[0]
            if (not callable(dt)) and dt > d_snap * (1.0 + 1e-10):
                warnings.warn(
                    f"dt = {dt:g} exceeds the snapshot spacing {d_snap:g}; "
                    f"reducing dt to {d_snap:g} so that every requested "
                    "snapshot corresponds to a distinct step.")
                dt = d_snap
            snap: Dict[str, list] = {k: [] for k in
                                     ("t", "q0", "q1", "dt", "gamma",
                                      "mass_tau", "step", "t_target")}

            def record(tt, dt_eff, g, target):
                q0s, q1s = self.unpack(U)
                snap["t"].append(tt);          snap["q0"].append(q0s)
                snap["q1"].append(q1s);        snap["dt"].append(dt_eff)
                snap["gamma"].append(g);       snap["step"].append(n)
                snap["t_target"].append(target)
                snap["mass_tau"].append(self.mass_tau(U))
                for key, val in self.diagnostics(U).items():
                    snap.setdefault(key, []).append(val)

            record(0.0, 0.0, 1.0, 0.0)         # the initial condition
            isnap = 1

        # times we are required to hit exactly
        if snap_t is not None and land_on_snapshots:
            land_times = list(snap_t[1:])
        elif land_on_T:
            land_times = [T]
        else:
            land_times = []
        il = 0

        def _sane(g):
            """Reject a nonsensical gamma (ill-conditioned for tiny dt)."""
            nonlocal n_clamped
            if not np.isfinite(g) or not (0.5 <= g <= 1.5):
                n_clamped += 1
                return 1.0
            return g

        dt_fn = dt if callable(dt) else None
        stopped = None
        while t < T - eps:
            if dt_fn is not None:
                dt = float(dt_fn(t, U, self))
                if not (dt > 0.0) or not np.isfinite(dt):
                    raise ValueError(f"dt(t={t:g}, ...) returned {dt}")
            # step size: plain dt, except when closing in on a landing time
            hit, horizon = False, None
            if il < len(land_times):
                horizon = land_times[il]
                remaining = horizon - t
                if remaining <= dt * (1.0 + 1e-10):
                    dt_n, hit = remaining, True        # land exactly
                elif remaining < 2.0 * dt:
                    dt_n = 0.5 * remaining             # avoid a tiny final step
                else:
                    dt_n = dt
            else:
                dt_n = dt

            if hit and relaxation and variant == "relaxation":
                # solve gamma(dt)*dt = horizon - t by fixed point (2-3 its)
                for _ in range(10):
                    d, g = self.step(U, dt_n, tab, relaxation=True)
                    g = _sane(g)
                    dt_new = remaining / g
                    conv = abs(dt_new - dt_n) <= 1e-14 * abs(dt_new)
                    dt_n = dt_new
                    if conv:
                        break
                d, g = self.step(U, dt_n, tab, relaxation=True)
                g = _sane(g)
            else:
                d, g = self.step(U, dt_n, tab, relaxation)
                if relaxation:
                    g = _sane(g)

            gmax = max(gmax, abs(g - 1.0))
            dt_eff = g * dt_n if variant == "relaxation" else dt_n
            U = U + (g * dt_n) * d
            t = t + dt_eff
            if hit and abs(t - horizon) < 1e-9 * max(1.0, abs(T)):
                t = horizon
                il += 1
            n += 1

            if callback is not None:
                callback(t, U, self)
            if store_every and (n % store_every == 0 or t >= T - eps):
                hist["t"].append(t)
                hist["dt"].append(dt_eff)
                hist["gamma"].append(g)
                hist["mass_tau"].append(self.mass_tau(U))
                for key, val in self.diagnostics(U).items():
                    hist[key].append(val)
            if snap_t is not None and isnap < snap_t.size \
                    and t >= snap_t[isnap] - eps:
                record(t, dt_eff, g, snap_t[isnap])
                while isnap < snap_t.size and t >= snap_t[isnap] - eps:
                    isnap += 1        # a long step may pass several targets
            if verbose and (dt_fn is not None or
                            n % max(1, (int(T / dt) // 10)) == 0):
                print(f"  step {n:6d}  t = {t:10.6f}  gamma-1 = {g-1:+.3e}  "
                      f"Mtau = {self.mass_tau(U):.15e}")
            if stop is not None and stop(t, U, self):
                stopped = "stop() predicate"
                break
            if n >= max_steps:
                stopped = f"max_steps = {max_steps}"
                break

        if stopped is not None and snap_t is not None \
                and snap["t"][-1] != t:
            record(t, hist["dt"][-1], hist["gamma"][-1], np.nan)  # keep the end
        if stopped is not None:
            warnings.warn(f"run stopped early at t = {t:g} < T = {T:g} "
                          f"({stopped}); snapshots are truncated there.")
        if snap_t is not None and stopped is None:   # safety net (rounding)
            while isnap < snap_t.size:
                record(t, hist["dt"][-1], hist["gamma"][-1], snap_t[isnap])
                isnap += 1

        if n_clamped:
            warnings.warn(f"{n_clamped} step(s): gamma outside [0.5, 1.5], "
                          "relaxation skipped there (gamma set to 1).")
        if relaxation and gmax > gamma_tol:
            warnings.warn(
                f"max |gamma-1| = {gmax:.2e} > {gamma_tol}: the semi-discrete "
                "flow is genuinely losing the invariant (under-resolved fast "
                "mode + L-stable damping?).  Relaxation is then papering over a "
                "real error - reduce dt, use 'cn_trap', or prepare the data.")

        snapshots = None
        if snap_t is not None:
            q1arr = (None if (len(snap["q1"]) == 0 or snap["q1"][0] is None)
                     else np.array(snap["q1"]))
            snapshots = Snapshots(
                t=np.asarray(snap["t"]), q0=np.array(snap["q0"]),
                q1=q1arr, dt=np.asarray(snap["dt"]),
                gamma=np.asarray(snap["gamma"]),
                mass_tau=np.asarray(snap["mass_tau"]),
                step=np.asarray(snap["step"], dtype=int),
                t_target=np.asarray(snap["t_target"]),
                diag={k: np.asarray(v) for k, v in snap.items()
                      if k not in ("t", "q0", "q1", "dt", "gamma",
                                   "mass_tau", "step", "t_target")})

        q0, q1 = self.unpack(U)
        return Result(t=t, q0=q0, q1=q1, U=U,
                      hist={k: np.asarray(v) for k, v in hist.items()},
                      snapshots=snapshots, nsteps=n, tableau=tab.name)



class NLSH1D(IMEXIntegrator):
    """Pseudospectral IMEX-ESDIRK integrator for the 1-D NLSH system.

    Parameters
    ----------
    M       : number of grid points (even, power of 2 preferred)
    L       : domain length; grid is [x0, x0+L)
    tau     : relaxation parameter (NLSH -> NLS as tau -> 0)
    kappa   : nonlinearity coefficient (>0 focusing with this sign convention)
    p       : nonlinearity exponent, f(q0) = kappa |q0|^{p-1} q0  (p = 3 cubic)
    V       : optional real potential, array of shape (M,) or callable V(x).
              Treated *explicitly* (it is non-stiff and mass-preserving).
    filter_kind, filter_alpha, filter_order : see `make_filter`
    """

    def __init__(self, M: int = 256, L: float = 2 * np.pi, x0: float = 0.0,
                 tau: float = 1e-2, kappa: float = 1.0, p: float = 3.0,
                 V=None, filter_kind: str = "2/3",
                 filter_alpha: float = 36.0, filter_order: int = 8,
                 tau_nls: float = 1e-14):
        self.grid = Fourier1D(M, L, x0)
        self.tau = float(tau)
        self.tau_nls = float(tau_nls)
        # --- automatic NLS fallback: for tau <= tau_nls the solver drops the
        #     second component and integrates NLS itself on the same grid -----
        self.mode = "nls" if self.tau <= self.tau_nls else "nlsh"
        self.ncomp = 1 if self.mode == "nls" else 2
        self.kappa = float(kappa)
        self.p = float(p)
        self.S = make_filter(self.grid, filter_kind, filter_alpha, filter_order)
        self.filter_kind = filter_kind
        if callable(V):
            V = V(self.grid.x)
        self.V = None if V is None else np.asarray(V, dtype=float)
        self._imp_cache: Dict[float, object] = {}

    def __repr__(self):
        return (f"NLSH1D(mode={self.mode}, M={self.grid.M}, L={self.grid.L:g}, "
                f"tau={self.tau:g}, kappa={self.kappa:g}, p={self.p:g}, "
                f"filter={self.filter_kind})")

    # -- transforms ----------------------------------------------------
    @property
    def x(self):
        return self.grid.x

    def to_fourier(self, q0, q1=None) -> np.ndarray:
        U = np.empty((self.ncomp, self.grid.M), dtype=complex)
        U[0] = np.fft.fft(np.asarray(q0, dtype=complex))
        if self.ncomp == 2:
            U[1] = np.fft.fft(np.asarray(q1, dtype=complex))
        return U

    def to_physical(self, U) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.mode == "nls":
            return np.fft.ifft(U[0]), None
        return np.fft.ifft(U[0]), np.fft.ifft(U[1])

    # IMEXIntegrator hooks
    def pack(self, q0, q1=None):
        if self.mode == "nls":
            if q1 is not None:
                warnings.warn("tau <= tau_nls: mode='nls', so the supplied q1 "
                              "is ignored (NLS is integrated directly).")
            return self.to_fourier(q0)
        if q1 is None:
            raise ValueError("NLSH mode needs both q0 and q1 "
                             "(use solver.well_prepared(q0) for q1 = d_x q0)")
        return self.to_fourier(q0, q1)

    def unpack(self, U):
        return self.to_physical(U)

    def diagnostics(self, U) -> Dict[str, float]:
        return {"mass_q0": self.mass_q0(U),
                "energy":  self.energy(U),
                "virial":  self.virial(U),
                "max_q0":  float(np.max(np.abs(np.fft.ifft(U[0]))))}

    def dx(self, q) -> np.ndarray:
        """Spectral derivative of a physical-space field (utility)."""
        return np.fft.ifft(1j * self.grid.k * np.fft.fft(q))

    # -- operators -----------------------------------------------------
    def Lop(self, U: np.ndarray) -> np.ndarray:
        """Stiff linear part, diagonal in Fourier space."""
        k = self.grid.k
        out = np.empty_like(U)
        if self.mode == "nls":
            out[0] = -1j * (k * k) * U[0]           # i d_xx  ->  -i k^2
            return out
        out[0] = -k * U[1]
        out[1] = (k / self.tau) * U[0] + (1j / self.tau) * U[1]
        return out

    def Nop(self, U: np.ndarray) -> np.ndarray:
        """Non-stiff nonlinear (+ potential) part; symmetric filtering."""
        q0 = np.fft.ifft(self.S * U[0])
        f = self.kappa * np.abs(q0) ** (self.p - 1.0) * q0
        if self.V is not None:
            f = f + self.V * q0
        out = np.zeros_like(U)
        out[0] = 1j * self.S * np.fft.fft(f)
        return out

    def solve_implicit(self, rhs: np.ndarray, a: float) -> np.ndarray:
        """Solve (I - a L) Y = rhs exactly, mode by mode.

        I - a M(k) = [[1, a k], [-a k/tau, 1 - i a/tau]],
        det = 1 - i a/tau + a^2 k^2/tau.
        """
        if a == 0.0:
            return rhs.copy()
        if self.mode == "nls":
            # L = -i k^2  =>  (I - a L)^{-1} = 1/(1 + i a k^2)
            inv = self._imp_cache.get(a)
            if inv is None:
                inv = 1.0 / (1.0 + 1j * a * self.grid.k ** 2)
                self._imp_cache[a] = inv
            out = np.empty_like(rhs)
            out[0] = inv * rhs[0]
            return out
        cached = self._imp_cache.get(a)
        if cached is None:
            k = self.grid.k
            if self.mode == "nls":
                cached = (1.0 / (1.0 + 1j * a * k * k),)
            else:
                tau = self.tau
                d11 = 1.0 - 1j * a / tau
                det = d11 + (a * a) * k * k / tau
                cached = (d11 / det, (-a * k) / det,
                          (a * k / tau) / det, 1.0 / det)
            self._imp_cache[a] = cached
        out = np.empty_like(rhs)
        if self.mode == "nls":
            out[0] = cached[0] * rhs[0]
            return out
        i00, i01, i10, i11 = cached
        out[0] = i00 * rhs[0] + i01 * rhs[1]
        out[1] = i10 * rhs[0] + i11 * rhs[1]
        return out

    def well_prepared(self, q0, order: int = 1) -> Optional[np.ndarray]:
        r"""q1 on the relaxation manifold to the given order in tau:

            order 1:  q1 = q0_x
            order 2:  q1 = q0_x + tau (q0_xx + f)_x,  f = (V + kappa|q0|^{p-1})q0

        Returns None in NLS mode.  Being spectral, order 2 is safe here (no FD
        amplification of the third derivative).
        """
        if self.mode == "nls":
            return None
        q0 = np.asarray(q0, dtype=complex)
        ik, k2 = 1j * self.grid.k, self.grid.k ** 2
        h0 = np.fft.fft(q0)
        q1 = np.fft.ifft(ik * h0)
        if order >= 2:
            f = self.kappa * np.abs(q0) ** (self.p - 1.0) * q0
            if self.V is not None:
                f = f + self.V * q0
            q1 = q1 + self.tau * np.fft.ifft(ik * (-k2 * h0 + np.fft.fft(f)))
        return q1

    def nonlinear_dt(self, dt_max: float, safety: float = 0.1,
                     floor: float = 0.0, quantize: bool = True):
        r"""Adaptive dt for the explicitly-treated nonlinearity.

        The explicit part has eigenvalues ~ i(kappa|q0|^{p-1} + V), sitting on
        the imaginary axis, so an s-stage ERK is stable only while

            dt * (kappa ||q0||_inf^{p-1} + ||V||_inf)  <~  1

        which is *the* constraint during blow-up: ||u||_inf -> infinity forces
        dt -> 0 like ||u||_inf^{-(p-1)}.  Returns a callable to hand to run().

        safety ~ 0.4 sits at the edge of stability but is far too coarse to be
        accurate; measured on a p=5 focusing run, the energy drift falls like
        safety^3 (the tableau order), so use ~0.1 or less and watch dE/E.

        quantize=True restricts dt to dt_max/2^j.  The implicit operator then
        only takes a handful of distinct values over a run, so its sparse LU is
        factorised once per level instead of once per step - a large saving when
        dt varies continuously.
        """
        Vmax = 0.0 if self.V is None else float(np.max(np.abs(self.V)))

        def dt_fn(t, U, solver):
            amp = float(np.max(np.abs(U[0])))
            rate = abs(self.kappa) * amp ** (self.p - 1.0) + Vmax
            d = dt_max if rate <= 0 else min(dt_max, safety / rate)
            d = max(d, floor)
            if quantize and d < dt_max:
                j = int(np.ceil(np.log2(dt_max / d)))
                d = max(dt_max * 2.0 ** (-j), floor)
            return d
        return dt_fn

    # -- inner products / invariants -----------------------------------
    def inner(self, U: np.ndarray, V: np.ndarray) -> float:
        """<U,V>_A = Re int (conj(u0) v0 + tau conj(u1) v1) dx  (via Parseval)."""
        w = self.grid.dx / self.grid.M
        s = np.vdot(U[0], V[0])
        if self.mode == "nlsh":
            s = s + self.tau * np.vdot(U[1], V[1])
        return float(w * np.real(s))

    def mass_tau(self, U) -> float:
        """The conserved mass  int |q0|^2 + tau |q1|^2 dx
        (just int |q0|^2 in NLS mode)."""
        return self.inner(U, U)

    def mass_q0(self, U) -> float:
        """int |q0|^2 dx  (the NLS mass; only conserved up to O(tau))."""
        w = self.grid.dx / self.grid.M
        return float(w * np.real(np.vdot(U[0], U[0])))

    def integrate(self, f) -> float:
        """dx * sum_j f_j  ~  int f dx (spectrally exact for band-limited f)."""
        return float(self.grid.dx * np.real(np.sum(f)))

    def energy(self, U) -> float:
        r"""NLSH energy, tracked in `hist["energy"]`:

            E_tau = int [ 2 Re(conj(q1) q0_x) - |q1|^2
                          - (2 kappa/(p+1))|q0|^{p+1} - V|q0|^2 ] dx
                  = E_NLS(q0) - || q0_x - q1 ||_{L^2}^2 ,

            E_NLS(q0) = int |q0_x|^2 - (2 kappa/(p+1))|q0|^{p+1} - V|q0|^2 .

        Conserved by NLSH; in NLS mode it is E_NLS itself.  The second form is
        what is evaluated.
        The -V|q0|^2 sign follows this system's convention, in which V enters
        the equation with the same sign as the nonlinearity.
        """
        q0 = np.fft.ifft(U[0])
        dq0 = np.fft.ifft(1j * self.grid.k * U[0])
        nl = (2.0 * self.kappa / (self.p + 1.0)) * np.abs(q0) ** (self.p + 1.0)
        if self.V is not None:
            nl = nl + self.V * np.abs(q0) ** 2
        e_nls = self.integrate(np.abs(dq0) ** 2 - nl)
        if self.mode == "nls":
            return e_nls
        q1 = np.fft.ifft(U[1])
        return e_nls - self.integrate(np.abs(dq0 - q1) ** 2)

    def virial(self, U) -> float:
        r"""Virial-like quantity, tracked in `hist["virial"]`:

            Jtilde_tau = J_tau + 4 tau B
                       = int |x|^2 (|q0|^2 + tau |q1|^2) dx
                         + 4 tau Re int conj(q0) (x . q1) dx ,

        which reduces to int x^2 |u|^2 in NLS mode.

        WARNING: |x|^2 is not periodic, so on this Fourier grid the quantity is
        meaningful only while the solution stays away from the boundaries, and
        only if the grid is centred on the origin - construct the solver with
        x0 = -L/2.
        """
        x = self.grid.x
        q0 = np.fft.ifft(U[0])
        J = self.integrate(x ** 2 * np.abs(q0) ** 2)
        if self.mode == "nls":
            return J
        q1 = np.fft.ifft(U[1])
        J += self.tau * self.integrate(x ** 2 * np.abs(q1) ** 2)
        J += 4.0 * self.tau * self.integrate(np.conj(q0) * x * q1)
        return J

# =====================================================================
# 4.  Exact plane-wave solution
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
    if tau <= 0.0:                       # NLS limit: omega = k^2 - kappa sigma^{p-1}
        if branch != "+":
            raise ValueError("the '-' (fast) branch does not exist at tau = 0")
        om = k * k - a
        q0 = sigma * np.exp(1j * (k * x - om * t))
        return q0, 1j * k * q0, om
    disc = np.sqrt(4.0 * tau * k * k + (a * tau - 1.0) ** 2)
    om = (-(1.0 + a * tau) + (1.0 if branch == "+" else -1.0) * disc) / (2.0 * tau)
    q0 = sigma * np.exp(1j * (k * x - om * t))
    q1 = (1j * k / (1.0 + tau * om)) * q0
    return q0, q1, om


def plane_wave_mass(L, k, sigma, kappa, p, tau, branch="+"):
    """Analytic value of int |q0|^2 + tau |q1|^2 dx for the plane wave."""
    _, _, om = plane_wave(0.0, 0.0, k, sigma, kappa, p, tau, branch)
    return L * sigma ** 2 * (1.0 + tau * (k / (1.0 + tau * om)) ** 2)


# =====================================================================
# 5.  Demos / tests
# =====================================================================

def _err(a, b):
    return float(np.max(np.abs(a - b)))


def plot_summary(sol: "NLSH1D", res: Result, exact=None, fname="nlsh_summary.png"):
    """Solution, invariant drift, relaxation parameter, and - if snapshots were
    stored - a space-time map of |q0|.  Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = res.hist
    s = res.snapshots
    ncol = 4 if s is not None else 3
    fig, ax = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.8))
    ax[0].plot(sol.x, np.abs(res.q0), lw=2, label=r"$|q_0|$")
    ax[0].plot(sol.x, np.real(res.q0), lw=1, alpha=.7, label=r"$\Re q_0$")
    if exact is not None:
        ax[0].plot(sol.x, np.abs(exact), "k--", lw=1, label="exact")
    ax[0].set_xlabel("x"); ax[0].set_title(f"t = {res.t:g}  ({res.tableau})")
    ax[0].legend(fontsize=8)

    d = np.abs(h["mass_tau"] - h["mass_tau"][0]) / abs(h["mass_tau"][0])
    ax[1].semilogy(h["t"], np.maximum(d, 1e-18), lw=1.5, label=r"$M_\tau$ (relaxed)")
    d0 = np.abs(h["mass_q0"] - h["mass_q0"][0]) / abs(h["mass_q0"][0])
    ax[1].semilogy(h["t"], np.maximum(d0, 1e-18), lw=1, alpha=.7,
                   label=r"$\int|q_0|^2$ (not invariant)")
    ax[1].set_xlabel("t"); ax[1].set_title("relative drift"); ax[1].legend(fontsize=8)

    ax[2].plot(h["t"][1:], h["gamma"][1:] - 1.0, lw=1.2)
    ax[2].set_xlabel("t"); ax[2].set_title(r"$\gamma - 1$")

    if s is not None:
        im = ax[3].pcolormesh(sol.x, s.t, s.abs_q0, shading="auto", cmap="magma")
        ax[3].plot(np.full(len(s), sol.x[0]), s.t, "w.", ms=3,
                   label="snapshot times")
        ax[3].set_xlabel("x"); ax[3].set_ylabel("t")
        ax[3].set_title(rf"$|q_0(x,t)|$, {len(s)} snapshots")
        ax[3].legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax[3])

    fig.tight_layout(); fig.savefig(fname, dpi=120)
    print(f"  wrote {fname}")
    return fname


def test_tableaux():
    print("=" * 74)
    print("TABLEAU VERIFICATION")
    print("=" * 74)
    ok = True
    for name in TABLEAUX:
        res = check_tableau(name)
        bad = {k: v for k, v in res.items()
               if k != "|R(inf)|" and abs(v) > 1e-12}
        ok &= not bad
        if bad:
            print("   !! failed:", bad)
        print()
    print("all tableaux:", "PASS" if ok else "FAIL")
    return ok


def test_plane_wave_residual(tau=0.05, k=3.0, sigma=0.8, kappa=1.5, p=3.0):
    """Check the analytic plane wave really solves the PDE (both branches)."""
    print("=" * 74)
    print("PLANE-WAVE RESIDUAL (symbolic check of the dispersion relation)")
    print("=" * 74)
    for br in ("+", "-"):
        _, _, om = plane_wave(0.0, 0.0, k, sigma, kappa, p, tau, br)
        a = kappa * sigma ** (p - 1)
        r_correct = (om + a) * (1 + tau * om) - k * k
        r_latex = (om + a) * (1 - tau * om) - k * k
        # residual of the PDE itself, evaluated on the grid
        sol = NLSH1D(M=32, L=2 * np.pi, tau=tau, kappa=kappa, p=p, filter_kind="none")
        q0, q1, _ = plane_wave(sol.x, 0.0, k, sigma, kappa, p, tau, br)
        U = sol.to_fourier(q0, q1)
        rhs = sol.Lop(U) + sol.Nop(U)
        dq0, dq1 = sol.to_physical(rhs)
        e = max(_err(dq0, -1j * om * q0), _err(dq1, -1j * om * q1))
        print(f"  branch {br}:  omega = {om:+.10f}   "
              f"|PDE residual| = {e:.3e}")
        print(f"             (om+a)(1+tau om) - k^2 = {r_correct:+.2e}   "
              f"[LaTeX's (1-tau om) form: {r_latex:+.2e}]")
    return True


def demo_plane_wave(method="ars443", tau=1e-2, dt=0.02, T=2.0, M=64,
                    k=2.0, sigma=1.0, kappa=1.0, p=3.0, branch="+"):
    print("=" * 74)
    print(f"DEMO 1 - plane wave, tableau = {method}, tau = {tau}, branch = {branch}")
    print("=" * 74)
    L = 2 * np.pi
    sol = NLSH1D(M=M, L=L, tau=tau, kappa=kappa, p=p, filter_kind="2/3")
    q0i, q1i, om = plane_wave(sol.x, 0.0, k, sigma, kappa, p, tau, branch)
    print(f"  omega_tau^{branch} = {om:.10f}   (NLS: omega = {k*k - kappa*sigma**(p-1):.10f})")
    print(f"  M = {M}, dt = {dt}, T = {T},  dt/tau = {dt/tau:.1f}")

    for relax in (False, True):
        r = sol.run(q0i, q1i, T, dt, method=method, relaxation=relax)
        q0e, q1e, _ = plane_wave(sol.x, r.t, k, sigma, kappa, p, tau, branch)
        m = r.hist["mass_tau"]
        print(f"  relaxation={str(relax):5s}: steps={r.nsteps:4d}  "
              f"err(q0)={_err(r.q0, q0e):.3e}  err(q1)={_err(r.q1, q1e):.3e}  "
              f"|dM/M|={np.max(np.abs(m - m[0]))/m[0]:.3e}  "
              f"max|gamma-1|={np.max(np.abs(r.hist['gamma']-1)):.2e}")
    print(f"  analytic mass = {plane_wave_mass(L,k,sigma,kappa,p,tau,branch):.15e}")
    return True


def demo_convergence(tau=1e-2, T=1.0, M=32, k=2.0, sigma=1.0, kappa=1.0, p=3.0):
    print("=" * 74)
    print(f"DEMO 2 - temporal convergence on the exact plane wave (tau = {tau})")
    print("=" * 74)
    sol = NLSH1D(M=M, L=2 * np.pi, tau=tau, kappa=kappa, p=p, filter_kind="2/3")
    q0i, q1i, _ = plane_wave(sol.x, 0.0, k, sigma, kappa, p, tau, "+")
    dts = [T / n for n in (20, 40, 80, 160, 320)]
    for relax in (False, True):
        print(f"\n  relaxation = {relax}")
        for name in TABLEAUX:
            tab = get_tableau(name)
            errs = []
            for dt in dts:
                r = sol.run(q0i, q1i, T, dt, method=name, relaxation=relax)
                q0e, _, _ = plane_wave(sol.x, r.t, k, sigma, kappa, p, tau, "+")
                errs.append(_err(r.q0, q0e))
            rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
            tag = "  [relaxation auto-off: gamma !-> 1 at order 1]" \
                  if (relax and tab.order < 2) else ""
            print(f"    {name:11s} (order {tab.order}): "
                  + "  ".join(f"{e:.2e}" for e in errs)
                  + "   rates: " + " ".join(f"{x:5.2f}" for x in rates) + tag)
    return True


def demo_stiff(method="ars443", dt=0.05, T=1.0):
    """dt >> tau, and the fast branch excited: the implicit part must cope."""
    print("=" * 74)
    print("DEMO 3 - stiff regime: fast (spurious) branch, dt/tau up to 1e4")
    print("=" * 74)
    for tau in (1e-2, 1e-3, 1e-4, 1e-5):
        sol = NLSH1D(M=32, L=2 * np.pi, tau=tau, kappa=1.0, p=3.0)
        q0i, q1i, om = plane_wave(sol.x, 0.0, 2.0, 1.0, 1.0, 3.0, tau, "-")
        r = sol.run(q0i, q1i, T, dt, method=method, relaxation=False, verbose=False)
        m = r.hist["mass_tau"]
        print(f"  tau={tau:7.0e}  omega^-={om:12.3e}  dt/tau={dt/tau:9.1f}  "
              f"|q0|_max={np.max(np.abs(r.q0)):.2e}  "
              f"M(T)/M(0)={m[-1]/m[0]:.2e}")
    print("  Stable for any dt/tau (unconditionally): the O(1/tau) mode sits in\n"
          "  the implicit part.  The L-stable tableaux annihilate it, which is\n"
          "  exactly the desired AP behaviour - but note M -> 0: relaxation must\n"
          "  NOT be switched on here, the invariant is being lost for a reason.")
    return True


def demo_soliton(tau=1e-3, T=2.0, dt=0.01, M=512, Lx=40.0, method="ars443"):
    """NLSH -> NLS: sech soliton of  i u_t + u_xx + 2|u|^2 u = 0,  u = sech(x)e^{it}."""
    print("=" * 74)
    print(f"DEMO 4 - sech soliton, NLSH -> NLS as tau -> 0 (T = {T})")
    print("=" * 74)
    for tau in (1e-2, 1e-3, 1e-4):
        sol = NLSH1D(M=M, L=Lx, x0=-Lx / 2, tau=tau, kappa=2.0, p=3.0,
                     filter_kind="2/3")
        q0i = 1.0 / np.cosh(sol.x)
        q1i = sol.dx(q0i)                     # well-prepared: q1 = d_x q0 + O(tau)
        r = sol.run(q0i, q1i, T, dt, method=method, relaxation=True)
        u_nls = np.exp(1j * r.t) / np.cosh(sol.x)
        m = r.hist["mass_tau"]
        print(f"  tau={tau:7.0e}:  ||q0 - u_NLS||_inf = {_err(r.q0, u_nls):.3e}   "
              f"|dM/M| = {np.max(np.abs(m-m[0]))/m[0]:.2e}   "
              f"max|gamma-1| = {np.max(np.abs(r.hist['gamma']-1)):.2e}")
    print("  (error should decrease ~linearly in tau, mass drift stays ~1e-16)")
    return True


def demo_snapshots(tau=1e-2, T=2.0, dt=0.013, n_snapshots=9, method="ars443"):
    """Store the full solution at n equally spaced output times."""
    print("=" * 74)
    print(f"DEMO 6 - {n_snapshots} snapshots, relaxation ON (dt = {dt}, not a "
          f"divisor of T = {T})")
    print("=" * 74)
    sol = NLSH1D(M=64, L=2 * np.pi, tau=tau, kappa=1.0, p=3.0)
    q0i, q1i, _ = plane_wave(sol.x, 0.0, 2.0, 1.0, 1.0, 3.0, tau, "+")

    for land in (False, True):
        r = sol.run(q0i, q1i, T, dt, method=method, relaxation=True,
                    n_snapshots=n_snapshots, land_on_snapshots=land)
        s = r.snapshots
        print(f"\n  land_on_snapshots = {land}   ({r.nsteps} steps, "
              f"q0 stored with shape {s.q0.shape})")
        print("    k   t_target        t          t-t_target      dt      "
              "gamma-1     M_tau              err(q0)")
        for i in range(len(s)):
            q0e, _, _ = plane_wave(sol.x, s.t[i], 2.0, 1.0, 1.0, 3.0, tau, "+")
            print(f"    {i}  {s.t_target[i]:8.4f}  {s.t[i]:11.7f}  "
                  f"{s.offsets[i]:+10.2e}  {s.dt[i]:8.6f}  {s.gamma[i]-1:+9.2e}  "
                  f"{s.mass_tau[i]:.14e}  {_err(s.q0[i], q0e):.2e}")
    print("\n  Either way the invariant is machine-exact at every snapshot; with\n"
          "  land_on_snapshots=False the output times drift by < 1 step, which is\n"
          "  the price of never touching the relaxation-chosen step.")
    return True


def demo_relaxation_longtime(tau=1e-2, T=50.0, dt=0.05, kappa=-1.0, p=3.0):
    """Defocusing (kappa < 0) so the dynamics stay smooth and resolved."""
    print("=" * 74)
    print(f"DEMO 5 - long-time mass drift, kappa = {kappa} (defocusing), T = {T}")
    print("=" * 74)
    sol = NLSH1D(M=128, L=2 * np.pi, tau=tau, kappa=kappa, p=p)
    q0i = 1.0 + 0.3 * np.cos(sol.x) + 0.2j * np.sin(2 * sol.x)
    q1i = sol.dx(q0i)                      # well-prepared data
    print(f"  {'tableau':12s} {'|dM/M| no relax':>16s} {'|dM/M| relax':>14s} "
          f"{'max|gamma-1|':>14s}")
    for name in TABLEAUX:
        if get_tableau(name).order < 2:
            continue                        # relaxation is meaningless at order 1
        row = [name]
        for relax in (False, True):
            r = sol.run(q0i, q1i, T, dt, method=name, relaxation=relax)
            m = r.hist["mass_tau"]
            row.append(np.max(np.abs(m - m[0])) / m[0])
            g = np.max(np.abs(r.hist["gamma"] - 1))
        print(f"  {row[0]:12s} {row[1]:16.2e} {row[2]:14.2e} {g:14.2e}")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("demo", nargs="?", default="all",
                    choices=["all", "tableaux", "residual", "planewave",
                             "convergence", "stiff", "soliton", "longtime",
                             "snapshots"])
    ap.add_argument("--method", default="ars443", choices=sorted(TABLEAUX))
    ap.add_argument("--tau", type=float, default=1e-2)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--plot", action="store_true", help="write nlsh_summary.png")
    ap.add_argument("--nsnap", type=int, default=120,
                    help="number of stored snapshots for --plot")
    a = ap.parse_args()

    if a.plot:
        sol = NLSH1D(M=256, L=40.0, x0=-20.0, tau=1e-3, kappa=2.0, p=3.0)
        q0i = 1.0 / np.cosh(sol.x)
        r = sol.run(q0i, sol.dx(q0i), a.T, a.dt, method=a.method, relaxation=True,
                    n_snapshots=a.nsnap)
        plot_summary(sol, r, exact=np.exp(1j * r.t) / np.cosh(sol.x))
        return

    if a.demo in ("all", "tableaux"):
        test_tableaux()
    if a.demo in ("all", "residual"):
        test_plane_wave_residual()
    if a.demo in ("all", "planewave"):
        demo_plane_wave(method=a.method, tau=a.tau, dt=a.dt, T=a.T)
    if a.demo in ("all", "convergence"):
        demo_convergence(tau=a.tau)
    if a.demo in ("all", "stiff"):
        demo_stiff(method=a.method)
    if a.demo in ("all", "soliton"):
        demo_soliton(method=a.method)
    if a.demo in ("all", "longtime"):
        demo_relaxation_longtime()
    if a.demo in ("all", "snapshots"):
        demo_snapshots(method=a.method)


if __name__ == "__main__":
    main()