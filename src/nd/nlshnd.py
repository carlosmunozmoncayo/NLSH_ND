r"""
nlshnd.py
=========
Fourier-pseudospectral / IMEX-ESDIRK solver for the N-dimensional first-order
hyperbolic approximation (NLSH) of the nonlinear Schroedinger equation.

This is the N-dimensional extension of `nlsh1d.py`, from which it imports the
*entire* time-stepping layer unchanged: the IMEX Butcher tableaux
(`TABLEAUX`, `IMEXTableau`, `check_tableau`), the relaxation/snapshot time loop
(`IMEXIntegrator.run`), and the `Result`/`Snapshots` containers.  Only the
space discretisation is new, so `NLSHND` with N = 1 reproduces `NLSH1D`
to machine precision (verified in `validate_nlshnd.py`).

Model (N dimensions)
--------------------
    i q0_t + div(q1) + (V(x) + kappa |q0|^{p-1}) q0 = 0
    i tau q1_t - grad(q0) + q1                      = 0

with q0 scalar and q1 = (q_1, ..., q_N) the vector field that approximates
grad(q0) in the relaxation limit, on a periodic box.  In evolution form

    q0_t = i div(q1) + i (kappa |q0|^{p-1} + V) q0
    q1_t = -(i/tau) grad(q0) + (i/tau) q1

split as  U_t = L(U) + N(U)  with

    L(U) = ( i div q1 ,  -(i/tau) grad q0 + (i/tau) q1 )  <- stiff, LINEAR -> implicit
    N(U) = ( i (kappa |q0|^{p-1} + V) q0 , 0 )            <- non-stiff     -> explicit

N = 1 is exactly the 1-D system of `nlsh1d.py` (div = grad = d_x).

Space
-----
Tensor-product Fourier pseudospectral, d_j -> i k_j, with the Nyquist mode of
each axis zeroed in that axis' derivative symbol.  L stays *diagonal in the
wavevector*: for each k = (k_1, ..., k_N)

    d/dt [q0h, q1h]^T = M(k) [q0h, q1h]^T ,
    M(k) = [[ 0        , -k^T        ],
            [ k/tau    , (i/tau) I_N ]]     ( (N+1) x (N+1) )

because div -> i k . and grad -> i k.  So every implicit stage is still an
*exact* solve per mode - no Newton, no Krylov - and the (N+1) x (N+1) system is
small enough to invert in closed form (see `solve_implicit`):

    (I - a M(k)) [y0, y1]^T = [r0, r1]^T
      =>  y0 = ( d11 r0 - a (k . r1) ) / det ,
          y1 = ( r1 + (a/tau) k y0 ) / d11 ,
    with  d11 = 1 - i a/tau ,  det = d11 + a^2 |k|^2 / tau .

Re(det) = 1 + a^2|k|^2/tau >= 1, so the solve is unconditionally well posed,
and for N = 1 the formulas collapse onto the 2x2 inverse used in `nlsh1d.py`.
Cost per stage: O(N) passes over the modes, no transforms (the state is held
in Fourier space throughout; FFTs appear only in the explicit nonlinearity).

Invariant
---------
    Mtau(U) = int |q0|^2 + tau |q1|^2 dx     ( |q1|^2 = sum_j |q_j|^2 )

is the modified mass of the paper.  L is skew-adjoint for the weighted inner
product
    <U,V>_A = Re int ( conj(q0) v0 + tau sum_j conj(q_j) v_j ) dx
- the two off-diagonal blocks -k^T and +k/tau are adjoint to each other up to
the tau weight - and N(U) is orthogonal to U in that product.  Both survive the
pseudospectral semi-discretisation exactly, so only the RK step breaks the
invariant, and `relaxation=True` restores it to machine precision exactly as in
1-D.

NLS fallback
------------
For tau <= tau_nls the solver drops the vector component and integrates

    i q0_t + Laplacian(q0) + (V + kappa|q0|^{p-1}) q0 = 0

directly: L = -i |k|^2, the implicit solve is a division by (1 + i a |k|^2).
|k|^2 is assembled from the *same* Nyquist-zeroed symbols as the first-order
operators, so NLS mode is the exact tau -> 0 limit of NLSH mode by mode, in
every dimension.

Diagnostics
-----------
    mass_q0 : int |q0|^2 dx                                  (drifts O(tau))
    energy  : Hbar = int |grad q0|^2 - |grad q0 - q1|^2
                         - (2 kappa/(p+1))|q0|^{p+1} - V|q0|^2 dx
              = H[q0] - || grad q0 - q1 ||_{L^2}^2 , the modified Hamiltonian;
              equals the NLS energy in NLS mode.  Method: `energy(U)`.
    virial  : int |x|^2 (|q0|^2 + tau|q1|^2) dx
              + 4 tau Re int conj(q0) (x . q1) dx .  Method: `virial(U)`.
              |x|^2 is not periodic: build with x0 = -L/2 and keep the solution
              away from the boundary, or drop it from `diag_keys`.
    max_q0  : ||q0||_inf

`momentum(U)` returns the N-vector of modified momenta
    Pbar_j = int Im(conj(q0) d_j q0) + tau sum_m Im(conj(q_m) d_j q_m) dx ,
which costs N(N+1) transforms and is therefore *not* in `diag_keys` by default.

Filtering
---------
The de-aliasing filter is the tensor product of the 1-D multipliers,
S(k) = prod_j s(k_j / kmax_j), applied symmetrically as N(U) = S f(S q0).  It
is real and separable, hence self-adjoint, so <U, N(U)>_A = 0 still holds and
filtering does not damage the conservation structure.  The '2/3' rule applied
per direction is the standard N-dimensional de-aliasing of a polynomial
nonlinearity.

Memory
------
State: (N+1) * prod(M) complex128.  The implicit solve caches one complex array
of shape prod(M) per distinct step size a (at most `cache_size`, LRU-evicted),
which matters in 3-D where each such array is as large as one state component.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import os
import sys

import numpy as np


thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(thisdir, '../1d'))

# The whole time-stepping layer is shared with the 1-D code.
from nlsh1d import (IMEXIntegrator, IMEXTableau, Result, Snapshots, TABLEAUX,
                    check_tableau, get_tableau, register_tableau,
                    stability_function)

__all__ = [
    "FourierND", "make_filter_nd", "NLSHND", "NLSH2D", "NLSH3D",
    # re-exported from nlsh1d for convenience
    "IMEXTableau", "TABLEAUX", "get_tableau", "check_tableau",
    "register_tableau", "stability_function", "Result", "Snapshots",
]


# =====================================================================
# 1.  Tensor-product Fourier grid
# =====================================================================

def _as_tuple(v, N, name, cast=float):
    """Broadcast a scalar or sequence to a length-N tuple."""
    arr = np.atleast_1d(np.asarray(v))
    if arr.size == 1:
        return tuple(cast(arr.flat[0]) for _ in range(N))
    if arr.size != N:
        raise ValueError(f"{name} has {arr.size} entries but the grid is "
                         f"{N}-dimensional")
    return tuple(cast(a) for a in arr.flat)


class FourierND:
    """Periodic box prod_j [x0_j, x0_j + L_j) with M_j points per direction.

    Attributes
    ----------
    N, shape      : dimension and (M_1, ..., M_N)
    dx, dV, Mtot  : grid spacings, cell volume prod(dx_j), prod(M_j)
    xs[j]         : 1-D coordinate array along axis j, shape (M_j,)
    X[j]          : the same, reshaped to broadcast against `shape`
    k_raw[j]      : broadcastable wavenumbers along axis j
    k[j]          : k_raw with the Nyquist mode zeroed (the derivative symbol)
    k2            : sum_j k[j]**2, full `shape` array (the Laplacian symbol)
    kmax[j]       : pi / dx_j
    axes          : (-N, ..., -1); the axes every transform acts on, so the
                    same call works for a scalar field (shape), a vector field
                    (N, *shape) and a full state (ncomp, *shape).
    """

    def __init__(self, M, L=2.0 * np.pi, x0: float = 0.0, N: Optional[int] = None):
        Marr = np.atleast_1d(np.asarray(M))
        if N is None:
            N = int(Marr.size) if Marr.size > 1 else 1
        self.N = int(N)
        self.shape = _as_tuple(M, self.N, "M", int)
        self.L = _as_tuple(L, self.N, "L", float)
        self.x0 = _as_tuple(x0, self.N, "x0", float)
        if any(m < 1 for m in self.shape):
            raise ValueError(f"every M_j must be >= 1, got {self.shape}")

        self.dx = tuple(self.L[j] / self.shape[j] for j in range(self.N))
        self.dV = float(np.prod(self.dx))
        self.Mtot = int(np.prod(self.shape))
        self.axes = tuple(range(-self.N, 0))

        self.xs, self.X, self.k_raw, self.k, self.kmax = [], [], [], [], []
        for j in range(self.N):
            Mj, dxj = self.shape[j], self.dx[j]
            bshape = [1] * self.N
            bshape[j] = Mj
            xj = self.x0[j] + dxj * np.arange(Mj)
            kj_raw = 2.0 * np.pi * np.fft.fftfreq(Mj, d=dxj)
            kj = kj_raw.copy()
            if Mj % 2 == 0:            # kill Nyquist in the odd-derivative symbol
                kj[Mj // 2] = 0.0
            self.xs.append(xj)
            self.X.append(xj.reshape(bshape))
            self.k_raw.append(kj_raw.reshape(bshape))
            self.k.append(kj.reshape(bshape))
            self.kmax.append(np.pi / dxj)

        # Laplacian symbol, built from the SAME zeroed k_j, so that NLS mode is
        # the exact tau -> 0 limit of NLSH mode by mode.
        self.k2 = np.zeros(self.shape)
        for j in range(self.N):
            self.k2 = self.k2 + self.k[j] ** 2

    def meshgrid(self) -> Tuple[np.ndarray, ...]:
        """Full-size coordinate arrays (indexing='ij').  For plotting only."""
        return tuple(np.broadcast_to(Xj, self.shape).copy() for Xj in self.X)

    def r2(self) -> np.ndarray:
        """|x|^2 = sum_j x_j^2, full `shape` array (used by `virial`)."""
        out = np.zeros(self.shape)
        for j in range(self.N):
            out = out + self.X[j] ** 2
        return out

    def __repr__(self):
        return (f"FourierND(N={self.N}, shape={self.shape}, "
                f"L={tuple(round(l, 6) for l in self.L)})")


def make_filter_nd(grid: FourierND, kind: str = "2/3",
                   alpha: float = 36.0, order: int = 8):
    """Real, separable Fourier multiplier S(k) = prod_j s(|k_j|/kmax_j).

    kind = 'none'   : S = 1  (returned as the scalar 1.0, costing no memory)
           '2/3'    : sharp cut at (2/3) kmax_j in every direction
           'exp'    : exp(-alpha sum_j (|k_j|/kmax_j)^(2*order))
           'hou-li' : exp(-36 sum_j (|k_j|/kmax_j)^36)

    Applied symmetrically in `Nop`, so self-adjointness - hence the exact
    orthogonality <U, N(U)>_A = 0 - is preserved.
    """
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return 1.0
    S = np.ones(grid.shape)
    for j in range(grid.N):
        kk = np.abs(grid.k_raw[j]) / grid.kmax[j]
        if kind in ("2/3", "23", "twothirds", "dealias"):
            Sj = (kk <= 2.0 / 3.0).astype(float)
        elif kind == "exp":
            Sj = np.exp(-alpha * kk ** (2 * order))
        elif kind in ("hou-li", "houli"):
            Sj = np.exp(-36.0 * kk ** 36)
        else:
            raise ValueError(f"unknown filter '{kind}'")
        S = S * Sj
    return S


# =====================================================================
# 2.  The N-dimensional solver
# =====================================================================

class NLSHND(IMEXIntegrator):
    """Pseudospectral IMEX-ESDIRK integrator for the N-dimensional NLSH system.

    Parameters
    ----------
    M        : points per direction - an int (same in every direction, then `N`
               must be given) or a sequence (M_1, ..., M_N), which sets N.
    L, x0    : box lengths and left corners; scalar (same in every direction)
               or a length-N sequence.
    N        : dimension; required only when M is a scalar.
    tau      : relaxation parameter (NLSH -> NLS as tau -> 0)
    kappa    : nonlinearity coefficient (>0 focusing with this sign convention)
    p        : nonlinearity exponent, f(q0) = kappa |q0|^{p-1} q0 (p=3 cubic)
    V        : optional real potential; an array broadcastable to `shape`, or a
               callable V(x_1, ..., x_N) of the N broadcastable coordinate
               arrays.  Treated explicitly (non-stiff and mass-preserving).
    diag_keys: which scalars `diagnostics` records each step.  'virial' is
               included by default to match the 1-D solver; drop it on large
               3-D runs (it needs an extra `shape`-sized real array).
    cache_size: how many distinct implicit step sizes to keep factorised.

    The state U is held in Fourier space with shape (1 + N, *shape) - component
    0 is q0h, components 1..N are the q1h_j - or (1, *shape) in NLS mode.
    Physical-space q1 has shape (N, *shape).
    """

    def __init__(self, M=32, L=2.0 * np.pi, x0: float = 0.0,
                 N: Optional[int] = None,
                 tau: float = 1e-2, kappa: float = 1.0, p: float = 3.0,
                 V=None, filter_kind: str = "2/3",
                 filter_alpha: float = 36.0, filter_order: int = 8,
                 tau_nls: float = 1e-14,
                 diag_keys: Sequence[str] = ("mass_q0", "energy",
                                             "virial", "max_q0"),
                 cache_size: int = 16):
        self.grid = FourierND(M, L, x0, N)
        self.N = self.grid.N
        self.tau = float(tau)
        self.tau_nls = float(tau_nls)
        # automatic NLS fallback, exactly as in 1-D
        self.mode = "nls" if self.tau <= self.tau_nls else "nlsh"
        self.ncomp = 1 if self.mode == "nls" else 1 + self.N
        self.kappa = float(kappa)
        self.p = float(p)
        self.S = make_filter_nd(self.grid, filter_kind, filter_alpha,
                                filter_order)
        self.filter_kind = filter_kind
        self.diag_keys = tuple(diag_keys)
        bad = set(self.diag_keys) - {"mass_q0", "energy", "virial", "max_q0"}
        if bad:
            raise ValueError(f"unknown diag_keys {sorted(bad)}")
        if callable(V):
            V = V(*self.grid.X)
        self.V = (None if V is None else
                  np.broadcast_to(np.asarray(V, dtype=float), self.grid.shape))
        self._imp_cache: Dict[float, tuple] = {}
        self._cache_size = int(cache_size)
        self._r2 = None                      # built lazily by `virial`

    def __repr__(self):
        return (f"NLSHND(N={self.N}, mode={self.mode}, "
                f"shape={self.grid.shape}, "
                f"L={tuple(round(l, 4) for l in self.grid.L)}, "
                f"tau={self.tau:g}, kappa={self.kappa:g}, p={self.p:g}, "
                f"filter={self.filter_kind})")

    # -- coordinates ---------------------------------------------------
    @property
    def x(self):
        """The 1-D coordinate array when N == 1 (as in NLSH1D), else the tuple
        of broadcastable coordinate arrays (x_1, ..., x_N)."""
        return self.grid.xs[0] if self.N == 1 else tuple(self.grid.X)

    @property
    def X(self) -> Tuple[np.ndarray, ...]:
        """The N broadcastable coordinate arrays, in every dimension."""
        return tuple(self.grid.X)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.grid.shape

    # -- transforms ----------------------------------------------------
    def _fftn(self, f):
        return np.fft.fftn(f, axes=self.grid.axes)

    def _ifftn(self, f):
        return np.fft.ifftn(f, axes=self.grid.axes)

    def to_fourier(self, q0, q1=None) -> np.ndarray:
        U = np.empty((self.ncomp, *self.grid.shape), dtype=complex)
        U[0] = self._fftn(np.asarray(q0, dtype=complex))
        if self.ncomp > 1:
            q1 = np.asarray(q1, dtype=complex)
            if q1.shape != (self.N, *self.grid.shape):
                raise ValueError(
                    f"q1 must have shape {(self.N, *self.grid.shape)}, "
                    f"got {q1.shape}")
            U[1:] = self._fftn(q1)
        return U

    def to_physical(self, U) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.mode == "nls":
            return self._ifftn(U[0]), None
        return self._ifftn(U[0]), self._ifftn(U[1:])

    # -- IMEXIntegrator hooks -----------------------------------------
    def pack(self, q0, q1=None):
        if self.mode == "nls":
            if q1 is not None:
                warnings.warn("tau <= tau_nls: mode='nls', so the supplied q1 "
                              "is ignored (NLS is integrated directly).")
            return self.to_fourier(q0)
        if q1 is None:
            raise ValueError("NLSH mode needs both q0 and q1 "
                             "(use solver.well_prepared(q0) for q1 = grad q0)")
        return self.to_fourier(q0, q1)

    def unpack(self, U):
        return self.to_physical(U)

    def diagnostics(self, U) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in self.diag_keys:
            if key == "mass_q0":
                out[key] = self.mass_q0(U)
            elif key == "energy":
                out[key] = self.energy(U)
            elif key == "virial":
                out[key] = self.virial(U)
            elif key == "max_q0":
                out[key] = float(np.max(np.abs(self._ifftn(U[0]))))
        return out

    # -- differential operators on physical fields (utilities) ---------
    def d(self, q, j: int) -> np.ndarray:
        """Spectral derivative d_j of a physical-space scalar field."""
        return self._ifftn(1j * self.grid.k[j] * self._fftn(q))

    def grad(self, q) -> np.ndarray:
        """grad q as an (N, *shape) physical-space field."""
        qh = self._fftn(np.asarray(q, dtype=complex))
        return np.stack([self._ifftn(1j * self.grid.k[j] * qh)
                         for j in range(self.N)])

    def div(self, qv) -> np.ndarray:
        """div of an (N, *shape) physical-space vector field."""
        qv = np.asarray(qv, dtype=complex)
        out = np.zeros(self.grid.shape, dtype=complex)
        for j in range(self.N):
            out = out + self._ifftn(1j * self.grid.k[j] * self._fftn(qv[j]))
        return out

    def lap(self, q) -> np.ndarray:
        """Laplacian of a physical-space scalar field (Nyquist-zeroed symbol)."""
        return self._ifftn(-self.grid.k2 * self._fftn(np.asarray(q, complex)))

    # keep the 1-D name working for N = 1
    def dx(self, q) -> np.ndarray:
        """d_x for N = 1 (alias of `d(q, 0)`); use `grad`/`d` in general."""
        return self.d(q, 0)

    # -- operators -----------------------------------------------------
    def Lop(self, U: np.ndarray) -> np.ndarray:
        """Stiff linear part; diagonal in the wavevector.

            (L U)_0   = -(k . q1h)
            (L U)_j+1 = (k_j/tau) q0h + (i/tau) q1h_j
        """
        out = np.empty_like(U)
        if self.mode == "nls":
            out[0] = -1j * self.grid.k2 * U[0]       # i Laplacian -> -i |k|^2
            return out
        k = self.grid.k
        acc = k[0] * U[1]
        for j in range(1, self.N):
            acc = acc + k[j] * U[1 + j]
        out[0] = -acc
        inv_tau, i_tau = 1.0 / self.tau, 1j / self.tau
        for j in range(self.N):
            out[1 + j] = (inv_tau * k[j]) * U[0] + i_tau * U[1 + j]
        return out

    def Nop(self, U: np.ndarray) -> np.ndarray:
        """Non-stiff nonlinear (+ potential) part; symmetric filtering."""
        q0 = self._ifftn(self.S * U[0])
        f = self.kappa * np.abs(q0) ** (self.p - 1.0) * q0
        if self.V is not None:
            f = f + self.V * q0
        out = np.zeros_like(U)
        out[0] = 1j * self.S * self._fftn(f)
        return out

    def _cache_put(self, a: float, value):
        if len(self._imp_cache) >= self._cache_size:
            self._imp_cache.pop(next(iter(self._imp_cache)))   # evict oldest
        self._imp_cache[a] = value
        return value

    def solve_implicit(self, rhs: np.ndarray, a: float) -> np.ndarray:
        r"""Solve (I - a L) Y = rhs exactly, wavevector by wavevector.

        I - a M(k) = [[ 1          ,  a k^T            ],
                      [ -a k/tau   ,  (1 - i a/tau) I  ]] .

        Eliminating y1 = (r1 + (a/tau) k y0)/d11 from the first row gives

            y0 = ( d11 r0 - a (k . r1) ) / det ,
            d11 = 1 - i a/tau ,   det = d11 + a^2 |k|^2 / tau ,

        and Re(det) = 1 + a^2|k|^2/tau >= 1, so nothing can be singular.  For
        N = 1 this is the 2x2 inverse of `nlsh1d.solve_implicit`.
        """
        if a == 0.0:
            return rhs.copy()
        out = np.empty_like(rhs)

        if self.mode == "nls":
            inv = self._imp_cache.get(a)
            if inv is None:
                inv = self._cache_put(a, 1.0 / (1.0 + 1j * a * self.grid.k2))
            out[0] = inv * rhs[0]
            return out

        cached = self._imp_cache.get(a)
        if cached is None:
            d11 = 1.0 - 1j * a / self.tau
            inv_det = 1.0 / (d11 + (a * a) * self.grid.k2 / self.tau)
            cached = self._cache_put(a, (d11, inv_det))
        d11, inv_det = cached

        k = self.grid.k
        kr = k[0] * rhs[1]
        for j in range(1, self.N):
            kr = kr + k[j] * rhs[1 + j]
        y0 = (d11 * rhs[0] - a * kr) * inv_det
        out[0] = y0
        c = (a / self.tau) / d11
        for j in range(self.N):
            out[1 + j] = rhs[1 + j] / d11 + (c * k[j]) * y0
        return out

    # -- data preparation ---------------------------------------------
    def well_prepared(self, q0, order: int = 1) -> Optional[np.ndarray]:
        r"""q1 on the relaxation manifold to the given order in tau:

            order 1:  q1 = grad q0
            order 2:  q1 = grad q0 + tau grad( Laplacian q0 + f ) ,
                      f = (V + kappa|q0|^{p-1}) q0 ,

        the second coming from q1 = grad q0 - i tau d_t(grad q0) + O(tau^2) and
        i d_t q0 = -(Laplacian q0 + f).  Returns an (N, *shape) array, or None
        in NLS mode.  Spectral differentiation makes order 2 safe (no FD
        amplification of the third derivatives).
        """
        if self.mode == "nls":
            return None
        q0 = np.asarray(q0, dtype=complex)
        h0 = self._fftn(q0)
        src = h0
        if order >= 2:
            f = self.kappa * np.abs(q0) ** (self.p - 1.0) * q0
            if self.V is not None:
                f = f + self.V * q0
            src = h0 + self.tau * (-self.grid.k2 * h0 + self._fftn(f))
        return np.stack([self._ifftn(1j * self.grid.k[j] * src)
                         for j in range(self.N)])

    def nonlinear_dt(self, dt_max: float, safety: float = 0.1,
                     floor: float = 0.0, quantize: bool = True):
        r"""Adaptive dt for the explicitly-treated nonlinearity.

        The explicit part has eigenvalues ~ i(kappa|q0|^{p-1} + V) on the
        imaginary axis, so the ERK is stable only while

            dt (kappa ||q0||_inf^{p-1} + ||V||_inf)  <~  1 ,

        which is the binding constraint during blow-up.  Returns a callable for
        `run(dt=...)`.  `quantize=True` restricts dt to dt_max/2^j so the
        implicit operator takes only a handful of distinct values and its
        per-mode factors are cached instead of rebuilt every step.

        NOTE: ||q0||_inf is measured in *physical* space (one inverse
        transform per step).  `nlsh1d.nonlinear_dt` takes the max of |U[0]|
        directly, but U is held in Fourier space, so that amplitude is too
        large by up to a factor prod(M) and the resulting dt is far smaller
        than the stability limit requires.
        """
        Vmax = 0.0 if self.V is None else float(np.max(np.abs(self.V)))

        def dt_fn(t, U, solver):
            amp = float(np.max(np.abs(self._ifftn(U[0]))))
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
        """<U,V>_A = Re int (conj(u0) v0 + tau sum_j conj(u_j) v_j) dx,
        evaluated by Parseval (the state lives in Fourier space)."""
        w = self.grid.dV / self.grid.Mtot
        s = np.vdot(U[0], V[0])
        if self.mode == "nlsh":
            s = s + self.tau * np.vdot(U[1:], V[1:])     # vdot flattens
        return float(w * np.real(s))

    def mass_tau(self, U) -> float:
        """int |q0|^2 + tau |q1|^2 dx (just int |q0|^2 in NLS mode)."""
        return self.inner(U, U)

    def mass_q0(self, U) -> float:
        """int |q0|^2 dx (the NLS mass; conserved only up to O(tau))."""
        w = self.grid.dV / self.grid.Mtot
        return float(w * np.real(np.vdot(U[0], U[0])))

    def integrate(self, f) -> float:
        """dV * sum f  ~  int f dx (spectrally exact for band-limited f)."""
        return float(self.grid.dV * np.real(np.sum(f)))

    def energy(self, U) -> float:
        r"""Modified Hamiltonian, tracked in `hist["energy"]`:

            Hbar = int [ 2 Re(conj(q1) . grad q0) - |q1|^2
                         - (2 kappa/(p+1))|q0|^{p+1} - V|q0|^2 ] dx
                 = H[q0] - || grad q0 - q1 ||_{L^2}^2 ,
            H[q0] = int |grad q0|^2 - (2 kappa/(p+1))|q0|^{p+1} - V|q0|^2 dx .

        The second form is what is evaluated.  In NLS mode it is H itself.
        The -V|q0|^2 sign follows this system's convention, in which V enters
        the equation with the same sign as the nonlinearity.
        """
        q0 = self._ifftn(U[0])
        nl = (2.0 * self.kappa / (self.p + 1.0)) * np.abs(q0) ** (self.p + 1.0)
        if self.V is not None:
            nl = nl + self.V * np.abs(q0) ** 2
        grad2 = np.zeros(self.grid.shape)
        defect = np.zeros(self.grid.shape)
        for j in range(self.N):
            dj = self._ifftn(1j * self.grid.k[j] * U[0])
            grad2 = grad2 + np.abs(dj) ** 2
            if self.mode == "nlsh":
                defect = defect + np.abs(dj - self._ifftn(U[1 + j])) ** 2
        e = self.integrate(grad2 - nl)
        if self.mode == "nls":
            return e
        return e - self.integrate(defect)

    def momentum(self, U) -> np.ndarray:
        r"""Modified momentum vector, Pbar_j for j = 1..N:

            Pbar_j = int Im(conj(q0) d_j q0)
                     + tau sum_m Im(conj(q_m) d_j q_m) dx ,

        which reduces to the NLS momentum int Im(conj(u) d_j u) in NLS mode.
        Costs N(N+1) inverse transforms, so it is a method rather than a
        per-step diagnostic; pass it through `run(callback=...)` if you want it
        recorded.
        """
        P = np.zeros(self.N)
        q0 = self._ifftn(U[0])
        comps = [q0]
        if self.mode == "nlsh":
            comps += [self._ifftn(U[1 + m]) for m in range(self.N)]
        for j in range(self.N):
            ikj = 1j * self.grid.k[j]
            acc = np.imag(np.conj(q0) * self._ifftn(ikj * U[0]))
            if self.mode == "nlsh":
                for m in range(self.N):
                    acc = acc + self.tau * np.imag(
                        np.conj(comps[1 + m]) * self._ifftn(ikj * U[1 + m]))
            P[j] = self.integrate(acc)
        return P

    def virial(self, U) -> float:
        r"""Virial-like quantity, tracked in `hist["virial"]`:

            Jtilde_tau = int |x|^2 (|q0|^2 + tau |q1|^2) dx
                         + 4 tau Re int conj(q0) (x . q1) dx ,

        reducing to int |x|^2 |u|^2 in NLS mode.

        WARNING: |x|^2 is not periodic, so on this Fourier grid the quantity is
        meaningful only while the solution stays away from the boundaries, and
        only if the box is centred on the origin - construct the solver with
        x0 = -L/2.
        """
        if self._r2 is None:
            self._r2 = self.grid.r2()
        q0 = self._ifftn(U[0])
        J = self.integrate(self._r2 * np.abs(q0) ** 2)
        if self.mode == "nls":
            return J
        cross = np.zeros(self.grid.shape, dtype=complex)
        for j in range(self.N):
            qj = self._ifftn(U[1 + j])
            J += self.tau * self.integrate(self._r2 * np.abs(qj) ** 2)
            cross = cross + self.grid.X[j] * qj
        J += 4.0 * self.tau * self.integrate(np.conj(q0) * cross)
        return J


def NLSH2D(M=32, L=2.0 * np.pi, x0: float = 0.0, **kw) -> NLSHND:
    """`NLSHND` with N = 2 (M, L, x0 may be scalars or length-2 sequences)."""
    return NLSHND(M, L, x0, N=2, **kw)


def NLSH3D(M=16, L=2.0 * np.pi, x0: float = 0.0, **kw) -> NLSHND:
    """`NLSHND` with N = 3 (M, L, x0 may be scalars or length-3 sequences)."""
    return NLSHND(M, L, x0, N=3, **kw)
