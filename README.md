# Simple Neural Networks Do Have the Universal Approximation Property

A constructive, runnable implementation of the paper

> **J. Kupka, Z. Alijani, P. Števuliáková** — *Simple Neural Networks Do Have Universal
> Approximation Property* (Institute for Research and Applications of Fuzzy Modeling, University of
> Ostrava).

The paper constructs a **strictly increasing, smooth, sigmoidal** activation function that
*encodes* a dense family of polynomials, so that structurally minimal feedforward networks become
universal approximators:

| Result | Network | Neurons | Weights |
|---|---|---|---|
| **Corollary 6.1** | one hidden layer, dim 1 | **2** | ±1 |
| **Theorem 4.2** | one hidden layer, ridge class M_n | 2n | unit vectors |
| **Corollary 6.2 / Thm 4.6** | two hidden layers, dim d | 2d then 4d+2 | fixed ±e_j |

This repo implements those constructions, verifies the key identities numerically, and compares the
constructed activation against standard ones.

## Contents

### Faithful implementation (recommended)
- **`UAP_implementation.ipynb`** — the paper's constructive core, section by section:
  Lemma 5.1 (exact `p = p⁺ − p⁻`), the dense monic sequence (Eq. 18), the construction of the
  activation κ with verification of its properties and the decoding identity (19), and the network
  theorems (Corollary 6.1, Theorem 4.2, Corollary 6.2).
- **`activation_networks.py`** — a reusable module that builds a one-hidden-layer network for
  **any** activation function (`build_one_hidden_layer`, `OneHiddenLayerNetwork`), plus the paper's
  constructive κ networks (`kappa_corollary_61`, `kappa_theorem_42`) and the full κ construction
  (`Kappa`, `monic_sequence`, `decompose_polynomial`). Run it directly for a comparison:
  `python activation_networks.py`.
- **`activation_comparison.ipynb`** — a visual comparison built on the module: the activation
  shapes, per-activation fits at a fixed neuron budget, and the decisive **error-vs-neurons**
  curves showing κ reaching with 2 neurons what standard activations need 32–128 for.

### Earlier exploratory demo
- `version1.ipynb`, `uap_constructive_demo.py`, `run_uap_demo.bat` — an initial "paper-inspired"
  activation and demo. Superseded by the faithful implementation above; kept for reference.

## Key numerical results (all verified)

| Quantity | Value |
|---|---|
| Lemma 5.1 reconstruction error | 6e-17 |
| κ strictly increasing / bounded in (−1, 1) | yes / yes |
| Decoding identity (19), worst over 40 polynomials | 2e-13 |
| Corollary 6.1 — **2 neurons**, target `sin(3x)exp(−0.3x)` | 6e-8 |
| Theorem 4.2 — **4 neurons** (2n, d=2) | 3e-9 |
| Corollary 6.2 — **4 + 10 neurons**, 2-D target | 6e-5 |

## Running

The code uses `numpy`, `scipy`, and `matplotlib`. Any recent scientific-Python environment works;
Jupyter is needed for the notebooks.

```bash
pip install numpy scipy matplotlib jupyter
python activation_networks.py          # comparison demo -> activation_comparison.png
jupyter lab UAP_implementation.ipynb   # the full construction
```

## Note

The paper's results are **purely theoretical** — the point is that a cleverly constructed activation
lets the network *structure* shrink to the minimum while retaining universal approximation. The
figures here illustrate and verify the constructions; they are not a claim about trainability.
