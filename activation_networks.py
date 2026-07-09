"""
activation_networks.py
======================

Build one-hidden-layer feedforward approximators for an *arbitrary* activation
function, so different activations can be compared on the same target.

The module offers two ways to "produce the NN":

1.  A **generic builder** `build_one_hidden_layer(...)` that works for *any*
    activation sigma.  It places hidden units across the input domain, then fits
    the output weights by least squares:
        F(x) = c0 + sum_i c_i * sigma(w_i . x + b_i).
    Use it with 'tanh', 'sigmoid', 'relu', 'sin', the paper's 'kappa', or your
    own callable.

2.  The paper's **constructive kappa networks** (Kupka, Alijani, Stevuliakova):
        * `kappa_corollary_61`  - 1-D, ONE hidden layer, exactly 2 neurons.
        * `kappa_theorem_42`    - ridge class M_n, ONE hidden layer, 2n neurons.
    These exploit the special structure of kappa and need far fewer neurons than
    a generic activation.

Run  `python activation_networks.py`  for a comparison demo.
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
from scipy.interpolate import PchipInterpolator

__all__ = [
    "relu", "logistic", "tanh", "sin", "gelu",
    "STANDARD_ACTIVATIONS", "get_activation",
    "Kappa", "make_kappa", "monic_sequence", "decompose_polynomial", "best_poly",
    "OneHiddenLayerNetwork", "build_one_hidden_layer",
    "kappa_corollary_61", "kappa_theorem_42",
]

# ===========================================================================
# 1. Standard activation functions (any callable R^. -> R^. works)
# ===========================================================================
def relu(x):     return np.maximum(0.0, x)
def logistic(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))   # sigmoid
def tanh(x):     return np.tanh(x)
def sin(x):      return np.sin(x)
def gelu(x):     return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

STANDARD_ACTIVATIONS = {
    "relu": relu, "sigmoid": logistic, "logistic": logistic,
    "tanh": tanh, "sin": sin, "gelu": gelu,
}


# ===========================================================================
# 2. The paper's constructed activation kappa  (Sections 3 & 6)
# ===========================================================================
def decompose_polynomial(p: Polynomial, slope: float = 1e-3):
    """Lemma 5.1 (exact): p = p_plus - p_minus with p_plus, p_minus strictly
    increasing on [-1, 1].  Returns two callables."""
    crit = np.sort([float(r.real) for r in np.atleast_1d(p.deriv().roots())
                    if abs(r.imag) < 1e-9 and -1.0 < r.real < 1.0])
    knots = np.concatenate(([-1.0], crit, [1.0]))
    pk = p(knots)
    inc = np.diff(pk) >= 0.0
    qp = np.zeros(len(knots)); qm = np.zeros(len(knots))
    qp[0] = float(p(-1.0))
    for i in range(len(knots) - 1):
        dpk = pk[i + 1] - pk[i]
        if inc[i]:  qp[i + 1] = qp[i] + dpk; qm[i + 1] = qm[i]
        else:       qp[i + 1] = qp[i];       qm[i + 1] = qm[i] - dpk

    def p_plus(x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
        return qp[i] + np.where(inc[i], p(x) - pk[i], 0.0) + slope * (x + 1.0)

    def p_minus(x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
        return qm[i] + np.where(inc[i], 0.0, pk[i] - p(x)) + slope * (x + 1.0)

    return p_plus, p_minus


def monic_sequence(count=48, max_degree=6, coeff_set=(0, 1, -1, 2, -2)):
    """A dense sequence of monic polynomials (paper Eq. 18), ordered by degree
    then coefficient complexity."""
    x = Polynomial([0, 1])
    seed = [Polynomial([1]), x**2, x, x**2 - x, x**2 - 1, x**3, x - 1, x**2 + x]
    seq, seen = [], set()
    def key(poly): return tuple(np.round(poly.coef, 6))
    def push(poly):
        if key(poly) not in seen:
            seen.add(key(poly)); seq.append(poly)
    for poly in seed: push(poly)
    for deg in range(max_degree + 1):
        combos = sorted(itertools.product(coeff_set, repeat=deg),
                        key=lambda t: (sum(abs(v) for v in t), t))
        for lowers in combos:
            push(Polynomial((list(lowers) + [1.0]) if deg else [1.0]))
            if len(seq) >= count: return seq
    return seq


class Kappa:
    """Strictly increasing, smooth, sigmoidal activation kappa: R -> (-1, 1)
    that encodes a list of polynomials.  Callable like any activation."""
    def __init__(self, polynomials, gamma=0.2, tail_scale=2.0, table_pts=8000):
        self.polys = list(polynomials); self.N = len(self.polys)
        self.tail_scale = tail_scale
        self.shift = [4 * n + 1 for n in range(1, self.N + 1)]       # d_n = 4n+1
        mus = np.linspace(0.05, 0.95, self.N)
        A = gamma * (mus[-1] - mus[0]) / max(self.N - 1, 1)
        self.pplus, self.pminus, self.dn, self.A, self.mu = [], [], [], A, mus
        for p in self.polys:
            pp, pm = decompose_polynomial(p)
            self.pplus.append(pp); self.pminus.append(pm)
            Mp = max(abs(float(pp(-1.0))), abs(float(pp(1.0))))
            Mm = max(abs(float(pm(-1.0))), abs(float(pm(1.0))))
            self.dn.append(A / max(Mp, Mm, 1e-9))
        self._build_table(table_pts)

    def _ln(self, i, x): return self.A * x + self.mu[i]
    def _enc_pos(self, i, s):
        x = s - self.shift[i]; return self.dn[i] * self.pplus[i](x) + self._ln(i, x)
    def _enc_neg(self, i, s):
        x = -s - self.shift[i]; return -(self.dn[i] * self.pminus[i](x) + self._ln(i, x))

    def _build_table(self, table_pts):
        ks, kv = [], []
        for i in reversed(range(self.N)):
            n = i + 1
            ks += [-4*n-2, -4*n]; kv += [float(self._enc_neg(i,-4*n-2)), float(self._enc_neg(i,-4*n))]
        for i in range(self.N):
            n = i + 1
            ks += [4*n, 4*n+2]; kv += [float(self._enc_pos(i,4*n)), float(self._enc_pos(i,4*n+2))]
        ks, kv = np.array(ks), np.array(kv); o = np.argsort(ks); ks, kv = ks[o], kv[o]
        self.s_min, self.s_max, self.v_min, self.v_max = ks[0], ks[-1], kv[0], kv[-1]
        grid = np.linspace(self.s_min, self.s_max, table_pts)
        self.table_s, self.table_v = grid, PchipInterpolator(ks, kv)(grid)

    def __call__(self, s):
        s = np.asarray(s, float); scalar = s.ndim == 0; s = np.atleast_1d(s)
        out = np.interp(np.clip(s, self.s_min, self.s_max), self.table_s, self.table_v)
        for i in range(self.N):
            n = i + 1
            m = (s >= 4*n) & (s <= 4*n+2)
            if np.any(m): out[m] = self._enc_pos(i, s[m])
            m = (s >= -4*n-2) & (s <= -4*n)
            if np.any(m): out[m] = self._enc_neg(i, s[m])
        lm = s < self.s_min
        if np.any(lm): out[lm] = self.v_min - (self.v_min+1)*(1-np.exp((s[lm]-self.s_min)/self.tail_scale))
        rm = s > self.s_max
        if np.any(rm): out[rm] = self.v_max + (1-self.v_max)*(1-np.exp(-(s[rm]-self.s_max)/self.tail_scale))
        return out[0] if scalar else out

    def decode(self, i, x):
        """Paper Eq. (19): recovers polynomial p_i(x) exactly."""
        a = 1.0 / self.dn[i]; d = self.shift[i]
        return a * self(-x - d) + a * self(x + d)

    def poly_index(self, p: Polynomial):
        """Append polynomial p to the encoded family and return its index."""
        self.polys.append(p); self.N += 1
        self.shift.append(4 * self.N + 1)
        self.mu = np.linspace(0.05, 0.95, self.N)
        self.A = 0.2 * (self.mu[-1] - self.mu[0]) / max(self.N - 1, 1)
        pp, pm = decompose_polynomial(p)
        self.pplus.append(pp); self.pminus.append(pm)
        Mp = max(abs(float(pp(-1.0))), abs(float(pp(1.0))))
        Mm = max(abs(float(pm(-1.0))), abs(float(pm(1.0))))
        # recompute all d_n because mu / A changed
        self.dn = [self.A / max(max(abs(float(a(-1.0))), abs(float(a(1.0)))),
                                max(abs(float(b(-1.0))), abs(float(b(1.0)))), 1e-9)
                   for a, b in zip(self.pplus, self.pminus)]
        self._build_table(8000)
        return self.N - 1


def make_kappa(count=40, **kw):
    """Convenience: kappa built from the first `count` monic polynomials."""
    return Kappa(monic_sequence(count=count), **kw)


def get_activation(name_or_callable):
    """Resolve 'tanh'/'sigmoid'/'relu'/'sin'/'gelu'/'kappa' or a callable."""
    if callable(name_or_callable):
        return name_or_callable
    key = str(name_or_callable).lower()
    if key == "kappa":
        return make_kappa()
    if key in STANDARD_ACTIVATIONS:
        return STANDARD_ACTIVATIONS[key]
    raise ValueError(f"unknown activation {name_or_callable!r}; "
                     f"known: {sorted(STANDARD_ACTIVATIONS) + ['kappa']}")


def best_poly(f, deg, grid=None):
    """Best degree-`deg` polynomial approximation of f on [-1, 1]."""
    if grid is None:
        grid = np.linspace(-1, 1, 600)
    return Chebyshev.fit(grid, f(grid), deg).convert(kind=Polynomial)


# ===========================================================================
# 3. Generic one-hidden-layer network for ANY activation
# ===========================================================================
def _as_X(X):
    """Coerce input to a 2-D design matrix (N, d).  A 1-D array is read as N
    samples of a single feature, i.e. shape (N, 1)."""
    X = np.asarray(X, float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class OneHiddenLayerNetwork:
    """F(x) = c0 + sum_i c_i * sigma(w_i . x + b_i).

    Hidden weights `W` (m, d) and biases `b` (m,) are fixed at construction;
    the output weights `c` (and bias `c0`) are obtained by least squares in
    `fit`.  Works for any activation callable."""
    def __init__(self, activation, W, b, use_bias=True):
        self.sigma = get_activation(activation)
        self.W = np.atleast_2d(np.asarray(W, float))
        self.b = np.asarray(b, float).ravel()
        self.use_bias = use_bias
        self.c = None; self.c0 = 0.0

    @property
    def n_neurons(self): return self.W.shape[0]

    def features(self, X):
        X = _as_X(X)
        Phi = self.sigma(X @ self.W.T + self.b)         # (N, m)
        if self.use_bias:
            Phi = np.column_stack([np.ones(len(X)), Phi])
        return Phi

    def fit(self, X, y):
        Phi = self.features(X)
        coef, *_ = np.linalg.lstsq(Phi, np.asarray(y, float).ravel(), rcond=None)
        if self.use_bias:
            self.c0, self.c = coef[0], coef[1:]
        else:
            self.c0, self.c = 0.0, coef
        return self

    def __call__(self, X):
        Phi = self.features(X)
        coef = np.concatenate([[self.c0], self.c]) if self.use_bias else self.c
        return Phi @ coef


def build_one_hidden_layer(activation, target, n_neurons, X_train, y_train=None,
                           domain=(-1.0, 1.0), steepness=None, seed=0):
    """Produce and fit a one-hidden-layer network for `activation`.

    * 1-D input: hidden units tile the domain (bias = -w * center), which suits
      sigmoidal / ReLU / periodic activations alike.
    * multi-D input: random Gaussian directions with biases spread over the
      projected range (random-features construction).

    Returns the fitted `OneHiddenLayerNetwork`."""
    X_train = _as_X(X_train)
    d = X_train.shape[1]
    if y_train is None:
        y_train = np.asarray(target(X_train if d > 1 else X_train.ravel())).ravel()
    rng = np.random.default_rng(seed)
    lo, hi = domain
    if d == 1:
        # spread centres across the domain, with random per-neuron frequencies so
        # the scheme also suits periodic activations (a single frequency would make
        # e.g. sine features rank-deficient).
        centers = np.linspace(lo, hi, n_neurons)
        base = (n_neurons / (hi - lo)) if steepness is None else steepness
        w = base * rng.uniform(0.3, 1.0, n_neurons) * rng.choice([-1.0, 1.0], n_neurons)
        W = w.reshape(-1, 1)
        b = -w * centers
    else:
        W = rng.normal(size=(n_neurons, d))
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
        w = (n_neurons ** (1.0 / d)) if steepness is None else steepness
        W *= w
        proj = X_train @ W.T
        b = -rng.uniform(proj.min(0), proj.max(0), size=n_neurons)
    net = OneHiddenLayerNetwork(activation, W, b).fit(X_train, y_train)
    return net


# ===========================================================================
# 4. Constructive kappa networks from the paper
# ===========================================================================
def kappa_corollary_61(target, degree=12, kappa=None, X_train=None):
    """Corollary 6.1: a ONE-hidden-layer, TWO-neuron network with activation
    kappa that approximates any f in C([-1,1]).

    Returns (predict_fn, info) where predict_fn(x) evaluates the 2-neuron net."""
    if X_train is None:
        X_train = np.linspace(-1, 1, 500)
    if kappa is None:
        kappa = Kappa(monic_sequence(count=8))
    idx = kappa.poly_index(best_poly(target, degree, grid=X_train))
    d = kappa.shift[idx]
    B = np.column_stack([kappa(-X_train - d), kappa(X_train + d)])   # 2 neurons
    c, *_ = np.linalg.lstsq(B, np.asarray(target(X_train)).ravel(), rcond=None)

    def predict(x):
        x = np.asarray(x, float)
        return c[0] * kappa(-x - d) + c[1] * kappa(x + d)

    err = float(np.max(np.abs(predict(X_train) - target(X_train))))
    return predict, {"n_neurons": 2, "weights": c, "shift": d,
                     "degree": degree, "max_error": err, "kappa": kappa}


def kappa_theorem_42(ridge_terms, directions, degree=12, kappa=None, X_train=None):
    """Theorem 4.2: a ONE-hidden-layer network with 2n neurons that approximates
    any f in the ridge class M_n = { sum_i g_i(a_i . x) }.

    Parameters
    ----------
    ridge_terms : list of n callables g_i : R -> R (univariate, on [-1, 1]).
    directions  : (n, d) array of unit vectors a_i in S^{d-1}.

    Each ridge term is decoded by 2 kappa-neurons whose weight vector is a_i
    (Corollary 6.1 along that direction), giving 2n neurons in one hidden layer.
    Returns (predict_fn, info)."""
    directions = np.atleast_2d(np.asarray(directions, float))
    ridge_terms = list(ridge_terms)
    assert len(ridge_terms) == directions.shape[0], "one ridge term per direction"
    if X_train is None:
        ax = np.linspace(-1, 1, 61)
        g = np.meshgrid(*([ax] * directions.shape[1]))
        P = np.column_stack([gg.ravel() for gg in g])
        X_train = P[(P ** 2).sum(1) <= 1.0]
    if kappa is None:
        kappa = Kappa(monic_sequence(count=8))
    target = lambda X: sum(g(X @ a) for g, a in zip(ridge_terms, directions))
    y = np.asarray(target(X_train)).ravel()

    # Encode every ridge polynomial FIRST (adding a polynomial rebalances the
    # value bands of all encoded polynomials), then build the basis once in the
    # final kappa state so fit-time and predict-time features coincide.
    meta = [(np.asarray(a, float), kappa.shift[kappa.poly_index(best_poly(g, degree))])
            for g, a in zip(ridge_terms, directions)]
    cols = []
    for a, d in meta:
        proj = X_train @ a
        cols += [kappa(-proj - d), kappa(proj + d)]                 # 2 neurons / dir
    B = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)

    def predict(X):
        X = _as_X(X)
        cc = []
        for a, d in meta:
            proj = X @ a
            cc += [kappa(-proj - d), kappa(proj + d)]
        return np.column_stack(cc) @ coef

    err = float(np.max(np.abs(predict(X_train) - y)))
    return predict, {"n_neurons": 2 * directions.shape[0], "max_error": err,
                     "kappa": kappa}


# ===========================================================================
# 5. Demo / comparison
# ===========================================================================
def _demo():
    import matplotlib.pyplot as plt

    target = lambda x: np.sin(3 * x) * np.exp(-0.3 * x)
    xs = np.linspace(-1, 1, 500)
    yt = target(xs)

    print("=" * 66)
    print("Approximating  f(x) = sin(3x) exp(-0.3x)  on [-1, 1]")
    print("=" * 66)

    # --- constructive kappa: only 2 neurons -------------------------------
    predict, info = kappa_corollary_61(target, degree=12)
    print(f"{'kappa (constructive, Cor 6.1)':32s}  neurons={info['n_neurons']:3d}"
          f"   max-err={info['max_error']:.2e}")

    # --- generic builder for several activations --------------------------
    results = {"kappa (2-neuron)": (2, info["max_error"], predict(xs))}
    for name in ["tanh", "sigmoid", "relu", "sin", "gelu"]:
        for m in [2, 8, 32, 128]:
            net = build_one_hidden_layer(name, target, m, xs)
            e = float(np.max(np.abs(net(xs) - yt)))
            if m == 32:
                results[f"{name} ({m} neurons)"] = (m, e, net(xs))
            print(f"{name + f' (generic, {m} neurons)':32s}  neurons={m:3d}"
                  f"   max-err={e:.2e}")
    print("-" * 66)
    print("Note: kappa reaches ~1e-7 with 2 neurons; standard activations need\n"
          "many neurons for comparable accuracy - the paper's whole point.")

    # --- plot -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, yt, "k", lw=2.5, label="target")
    for label, (m, e, yp) in results.items():
        ax.plot(xs, yp, lw=1.5, label=f"{label}  (err {e:.1e})")
    ax.set_title("One-hidden-layer approximation for different activations")
    ax.set_xlabel("x"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    out = "activation_comparison.png"
    fig.savefig(out, dpi=140)
    print(f"\nfigure saved to {out}")


if __name__ == "__main__":
    _demo()
