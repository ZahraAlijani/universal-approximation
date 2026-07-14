"""
kappa_h_compare.py
==================

Build the paper's activation kappa using the *explicit* odd sigmoidal scaffold
h(t) from page 11 of UAP_the_lastest.pdf:

        h(t) = 2 * ( 1 / (1 + e^{-t}) - 1/2 )   =   2*sigma(t) - 1   =   tanh(t/2)

Page 11 constructs kappa as h with *small perturbations*: on each window
[4n, 4n+2] the value follows an encoded polynomial band d_n * p_n^+(t) + l_n(t),
and everywhere else kappa == h.  Choosing the band offset l_n = h(4n+1) pins the
band onto h's own level, so h supplies every connector between bands AND both
+/-1 tails (unlike the PCHIP-scaffold version in activation_networks.py).

We then run Corollary 6.1 (a ONE-hidden-layer, TWO-neuron kappa network) on
    f1(x) = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720   (deg-6 Taylor of e^x)
    f2(x) = 4x / (4 + x^2)
and compare against the original PCHIP kappa and a plain tanh 2-neuron net.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial import Polynomial

from activation_networks import (
    decompose_polynomial, best_poly,
    Kappa, kappa_corollary_61, build_one_hidden_layer,
)


# --------------------------------------------------------------------------
# The page-11 odd sigmoidal scaffold h
# --------------------------------------------------------------------------
def h(t):
    """h(t) = 2*(1/(1+e^{-t}) - 1/2) = tanh(t/2).  Odd, C^inf, strictly
    increasing, sigmoidal:  h(+inf)=1, h(-inf)=-1."""
    t = np.asarray(t, float)
    return 2.0 / (1.0 + np.exp(-np.clip(t, -60, 60))) - 1.0


class KappaH:
    """kappa built ON TOP of the explicit odd sigmoidal h from page 11.

    kappa(s) = h(s) everywhere, except on the encoding windows
        [4n, 4n+2]      : kappa(s) = d_n * p_n^+(s-(4n+1)) + h(4n+1)
        [-4n-2, -4n]    : kappa(s) = -( d_n * p_n^-(-s-(4n+1)) + h(4n+1) )
    so that (Eq. 19)   (1/d_n) * ( kappa(x+4n+1) + kappa(-x-4n-1) ) = p_n(x)
    exactly.  The band amplitude d_n is kept inside the slot
        d_n * max|p_n^+/-| <= rho * ( h(4n+2) - h(4n) )
    so kappa stays a *small perturbation* of h (page 17, Question 3)."""

    def __init__(self, polynomials, rho=0.3):
        self.polys = list(polynomials)
        self.N = len(self.polys)
        self.rho = rho
        self.shift, self.pplus, self.pminus, self.dn, self.off = [], [], [], [], []
        for i, p in enumerate(self.polys):
            self._add(p)

    def _add(self, p):
        i = len(self.shift)
        n = i + 1
        d = 4 * n + 1
        pp, pm = decompose_polynomial(p)
        M = max(abs(float(pp(-1.0))), abs(float(pp(1.0))),
                abs(float(pm(-1.0))), abs(float(pm(1.0))), 1e-12)
        slot = float(h(4 * n + 2) - h(4 * n))          # width of the h-band
        amp = self.rho * slot                          # keep band inside the slot
        self.shift.append(d)
        self.pplus.append(pp); self.pminus.append(pm)
        self.dn.append(amp / M)
        self.off.append(float(h(d)))                   # l_n = h(4n+1): band rides on h

    def _enc_pos(self, i, s):
        x = s - self.shift[i]
        return self.dn[i] * self.pplus[i](x) + self.off[i]

    def _enc_neg(self, i, s):
        x = -s - self.shift[i]
        return -(self.dn[i] * self.pminus[i](x) + self.off[i])

    def __call__(self, s):
        s = np.asarray(s, float)
        scalar = s.ndim == 0
        s = np.atleast_1d(s)
        out = h(s)                                     # <-- the scaffold everywhere
        for i in range(len(self.shift)):
            n = i + 1
            m = (s >= 4 * n) & (s <= 4 * n + 2)
            if np.any(m): out[m] = self._enc_pos(i, s[m])
            m = (s >= -4 * n - 2) & (s <= -4 * n)
            if np.any(m): out[m] = self._enc_neg(i, s[m])
        return out[0] if scalar else out

    def decode(self, i, x):
        """Eq. (19): recovers p_i(x) exactly, independent of the offset h(4n+1)."""
        a = 1.0 / self.dn[i]; d = self.shift[i]
        return a * self(-x - d) + a * self(x + d)

    def poly_index(self, p):
        """Append polynomial p; other windows are untouched (h-mode is local)."""
        self.polys.append(p); self.N += 1
        self._add(p)
        return self.N - 1


def kappa_corollary_61_h(target, degree=12, kappa=None, X_train=None):
    """Corollary 6.1 with the h-scaffolded kappa: a ONE-hidden-layer, TWO-neuron
    network F(x) = c0*kappa(-x-d) + c1*kappa(x+d) that approximates target."""
    if X_train is None:
        X_train = np.linspace(-1, 1, 500)
    if kappa is None:
        # encode the target at window n=1 (shift 5): analytically any window works,
        # but numerically only the low ones survive -- h(4n+1) saturates to 1 so
        # fast that for n>=~4 the band amplitude drops below float64 resolution of
        # 1.0 and the polynomial information is lost (1.0 + 1e-16 == 1.0).
        kappa = KappaH([])
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


# --------------------------------------------------------------------------
# Targets and comparison
# --------------------------------------------------------------------------
def f1(x):   # degree-6 Taylor polynomial of e^x
    x = np.asarray(x, float)
    return 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + x**6/720

def f2(x):   # rational, odd
    x = np.asarray(x, float)
    return 4.0 * x / (4.0 + x**2)


def _metrics(pred, target, xs):
    e = pred(xs) - target(xs)
    return float(np.max(np.abs(e))), float(np.sqrt(np.mean(e**2)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.linspace(-1, 1, 2000)
    targets = [("f1(x)=1+x+x^2/2+...+x^6/720", f1, 6),
               ("f2(x)=4x/(4+x^2)",            f2, 14)]

    print("=" * 74)
    print("h-scaffolded kappa   h(t) = 2*(1/(1+e^-t) - 1/2) = tanh(t/2)")
    print("Corollary 6.1: ONE hidden layer, TWO neurons")
    print("=" * 74)
    print(f"{'target':34s} {'method':22s} {'neurons':>7s} {'max-err':>10s} {'rms-err':>10s}")
    print("-" * 74)

    results = {}
    for name, f, deg in targets:
        # (a) h-scaffolded kappa, 2 neurons
        pr_h, info_h = kappa_corollary_61_h(f, degree=deg)
        mh, rh = _metrics(pr_h, f, xs)
        # (b) original PCHIP kappa, 2 neurons (for reference)
        pr_k, info_k = kappa_corollary_61(f, degree=deg)
        mk, rk = _metrics(pr_k, f, xs)
        # (c) plain tanh, 2 neurons (baseline)
        net = build_one_hidden_layer("tanh", f, 2, xs)
        mt, rt = _metrics(net, f, xs)

        print(f"{name:34s} {'kappa_h (2n)':22s} {2:7d} {mh:10.2e} {rh:10.2e}")
        print(f"{'':34s} {'kappa PCHIP (2n)':22s} {2:7d} {mk:10.2e} {rk:10.2e}")
        print(f"{'':34s} {'tanh (2n)':22s} {2:7d} {mt:10.2e} {rt:10.2e}")
        print("-" * 74)
        results[name] = dict(f=f, pr_h=pr_h, mh=mh, pr_k=pr_k, net=net, deg=deg)

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

    for r, (name, f, deg) in enumerate(targets):
        R = results[name]
        axf = fig.add_subplot(gs[r, 0])
        axf.plot(xs, f(xs), "k", lw=2.4, label="target")
        axf.plot(xs, R["pr_h"](xs), "C0", lw=1.4, label=f"kappa_h 2-neuron")
        axf.plot(xs, R["net"](xs), "C3", lw=1.0, ls=":", label="tanh 2-neuron")
        axf.set_title(name); axf.set_xlabel("x"); axf.grid(alpha=.3); axf.legend(fontsize=8)

        axe = fig.add_subplot(gs[r, 1])
        axe.semilogy(xs, np.abs(R["pr_h"](xs) - f(xs)) + 1e-18, "C0", lw=1.2,
                     label=f"kappa_h  (max {R['mh']:.1e})")
        axe.semilogy(xs, np.abs(R["net"](xs) - f(xs)) + 1e-18, "C3", lw=1.0, ls=":",
                     label="tanh")
        axe.set_title(f"pointwise |error|  (degree {deg})")
        axe.set_xlabel("x"); axe.grid(alpha=.3, which="both"); axe.legend(fontsize=8)

    out = "kappa_h_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
