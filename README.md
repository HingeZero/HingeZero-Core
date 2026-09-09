# HingeZero Core

HingeZero is a deterministic associative-memory mechanism built around a
**hinge-at-zero nonlinearity**.

The core operator is:

\[
\phi(h;\alpha)=\tanh(h)+\alpha\tanh(2h)
\]

with iterative recall:

\[
h_t = Wx_t
\]

\[
x_{t+1}=(1-\lambda)x_t+\epsilon\phi(h_t;\alpha)
\]

The implementation is intentionally minimal, transparent, and reproducible.

---

## Overview

HingeZero treats the neighbourhood around zero as an **active equilibrium
region** rather than forcing every weak or ambiguous state immediately toward
a hard positive or negative decision.

The nonlinear operator is:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

where:

- `h` is the local memory field
- `alpha` controls the contribution of the second nonlinear term
- `φ(h; α)` is the HingeZero response

During associative recall, the local field is obtained from:

```text
h = W @ x
```

and the state is updated by:

```text
x_next = (1 - lam) * x + eps * φ(h; α)
```

Therefore the complete HingeZero recurrence is:

```text
h_t     = W @ x_t

φ_t     = tanh(h_t) + α·tanh(2h_t)

x_{t+1} = (1 - λ)x_t + εφ_t
```

---

## Canonical HingeZero Definition

Throughout this repository, the canonical HingeZero nonlinearity is:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

The canonical recall update is:

```text
x_{t+1} = (1 - λ)x_t + ε·φ(Wx_t; α)
```

Default parameters used by the core implementation are:

```text
α = 0.25
ε = 0.10
λ = 0.02
steps = 2
```

These defaults are reference values rather than requirements.

---

## Core Implementation

```python
import numpy as np


def hinge_phi(h, alpha=0.25):
    """
    Canonical HingeZero nonlinearity.

    φ(h; α) = tanh(h) + α·tanh(2h)
    """
    return np.tanh(h) + alpha * np.tanh(2.0 * h)


def hz_recall(
    W,
    x0,
    steps=2,
    alpha=0.25,
    eps=0.10,
    lam=0.02,
):
    """
    Perform deterministic HingeZero associative recall.

    h_t     = W @ x_t
    φ_t     = φ(h_t; α)
    x_{t+1} = (1 - λ)x_t + εφ_t
    """

    x = np.asarray(x0, dtype=float).copy()

    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1.0 - lam) * x + eps * phi

    return x
```

The mathematical operator and Python implementation above should be treated as
the reference definition of HingeZero Core.

---

## Key Properties

- **Deterministic recall**  
  Given the same memory matrix, initial state, parameters, and numerical
  environment, HingeZero performs the same update sequence.

- **Nonlinear associative update**  
  Recall is driven by the nonlinear response of the memory field rather than
  by a hard threshold alone.

- **Near-zero sensitivity**  
  The operator retains a continuous response around zero rather than imposing
  an immediate binary collapse.

- **Minimal implementation**  
  The essential recall mechanism requires only a small amount of code.

- **Reproducible**  
  The core recall function contains no random sampling or hidden stochastic
  state.

- **Quantisation compatible**  
  HingeZero can be investigated with reduced-precision and quantised memory
  representations, although quantisation behaviour depends on the surrounding
  implementation.

- **Easy to integrate**  
  The reference implementation requires only NumPy.

---

## Why the Hinge?

The name **HingeZero** refers to the behaviour of the nonlinear response around
the zero-field region.

The operator combines two bounded nonlinear components:

```text
tanh(h)
```

and:

```text
α·tanh(2h)
```

giving:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

The second term modifies the response around the origin while preserving a
bounded nonlinear structure.

For the default:

```text
α = 0.25
```

the operator becomes:

```text
φ(h) = tanh(h) + 0.25·tanh(2h)
```

---

## Associative Recall

HingeZero itself does not define how the memory matrix `W` must be constructed.

Different associative-memory schemes can provide `W`.

Once `W` exists, HingeZero acts on the field:

```text
h = W @ x
```

and applies the HingeZero response before updating the current state.

This separation is intentional:

```text
memory construction  →  W
query/state           →  x
memory field          →  h = W @ x
HingeZero response    →  φ(h; α)
state update          →  x_next
```

---

## Minimal Example

The following example constructs a simple associative matrix from stored
patterns and then recalls a perturbed query.

```python
import numpy as np


def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha * np.tanh(2.0 * h)


def hz_recall(
    W,
    x0,
    steps=2,
    alpha=0.25,
    eps=0.10,
    lam=0.02,
):
    x = np.asarray(x0, dtype=float).copy()

    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1.0 - lam) * x + eps * phi

    return x


# ------------------------------------------------------------
# Example stored patterns
# ------------------------------------------------------------

patterns = np.array([
    [ 1,  1, -1, -1,  1, -1],
    [-1,  1,  1, -1, -1,  1],
], dtype=float)


# ------------------------------------------------------------
# Simple associative memory matrix
# ------------------------------------------------------------

W = patterns.T @ patterns

np.fill_diagonal(W, 0.0)

W /= patterns.shape[1]


# ------------------------------------------------------------
# Perturbed query
# ------------------------------------------------------------

x0 = np.array(
    [1.0, 0.6, -1.0, -0.5, 1.0, -1.0],
    dtype=float,
)


# ------------------------------------------------------------
# HingeZero recall
# ------------------------------------------------------------

recalled = hz_recall(
    W,
    x0,
    steps=5,
    alpha=0.25,
    eps=0.10,
    lam=0.02,
)

print("Initial:")
print(x0)

print("\nRecalled:")
print(recalled)
```

This example is deliberately small and is intended to demonstrate the recall
mechanism rather than establish performance claims.

---

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `alpha` | `0.25` | Weight of the `tanh(2h)` component |
| `eps` | `0.10` | HingeZero update magnitude |
| `lam` | `0.02` | Retention / decay coefficient |
| `steps` | `2` | Number of iterative recall updates |

### Alpha

```text
α
```

controls the contribution of the second nonlinear component:

```text
α·tanh(2h)
```

The canonical operator remains:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

### Epsilon

```text
ε
```

controls how strongly the HingeZero response contributes during each update.

### Lambda

```text
λ
```

controls how much of the previous state is retained through:

```text
(1 - λ)x_t
```

### Steps

`steps` determines how many times the deterministic update is applied.

---

## Determinism

The HingeZero recall operation contains no intrinsic randomness.

For fixed:

```text
W
x0
α
ε
λ
steps
```

the update trajectory is deterministic, subject to the usual numerical
differences that can arise between hardware, floating-point formats, and
software implementations.

Randomness used to generate experimental datasets, initialise test memories,
or corrupt benchmark queries is separate from the HingeZero recall operator.

---

## Quantisation

The HingeZero architecture can be combined with low-bit representations.

Possible implementations include:

```text
FP32
FP16
INT8
binary / 1-bit storage
packed-bit memory representations
```

The nonlinear reference implementation shown in this repository uses NumPy
floating-point arithmetic.

Quantised variants may approximate, transform, or separate parts of the
calculation depending on their architecture.

See the dedicated HingeZero 1-bit work for specialised implementations.

---

## Design Philosophy

HingeZero Core follows several principles:

### Minimal

The core mechanism should remain small enough to inspect directly.

### Explicit

The operator and update equations should be visible rather than hidden behind
a large framework.

### Deterministic

Associative recall should not require stochastic sampling.

### Modular

Memory construction and HingeZero recall are separate components.

### Reproducible

The equations, default parameters, and implementation should correspond
directly.

---

## Notation Standard

To prevent ambiguity across implementations and publications, HingeZero uses
the following notation:

```text
x_t        current state
W          associative memory matrix
h_t        local memory field
α          hinge weighting parameter
ε          update magnitude
λ          state retention / decay parameter
φ          HingeZero nonlinear operator
```

Canonical equations:

```text
h_t = W @ x_t
```

```text
φ(h_t; α) = tanh(h_t) + α·tanh(2h_t)
```

```text
x_{t+1} = (1 - λ)x_t + ε·φ(h_t; α)
```

Equivalent compact form:

```text
x_{t+1} =
    (1 - λ)x_t
    + ε·φ(Wx_t; α)
```

---

## Repository Scope

This repository contains the **HingeZero Core** only.

It is intended to provide:

- the canonical operator
- the canonical recall equation
- the minimal Python implementation
- parameter definitions
- implementation guidance
- mathematical notation

Experimental wrappers, specialised memory architectures, tracking systems,
large-scale retrieval experiments, and application-specific implementations
should remain separate from the core definition.

This keeps the reference implementation stable.

---

## Related Implementations

### HingeZero.py

Full Python implementation:

https://github.com/HingeZero/HingeZero.py

### HingeZero-1Bit

Low-bit / packed representation work:

https://github.com/HingeZero/HingeZero-1Bit

---

## Citation

If you use HingeZero Core in research, please cite:

```bibtex
@software{hingezero2026,
  author = {HingeZero},
  title = {HingeZero Core: Deterministic Associative Memory},
  url = {https://github.com/HingeZero/HingeZero-Core},
  year = {2026}
}
```

---

## License

GNU General Public License v3.0.

See:

```text
LICENSE
```

for the complete license terms.

---

## Documentation

See:

- `CORE_IMPLEMENTATION.md` — mathematical and implementation details
- `HingeZero.py` — extended Python implementation
- `HingeZero-1Bit` — specialised low-bit implementation

---

## Canonical Reference

For consistency, all HingeZero implementations should trace their nonlinear
operator back to:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

and their standard iterative recall dynamics to:

```text
h_t = W @ x_t
x_{t+1} = (1 - λ)x_t + ε·φ(h_t; α)
```

Unless an implementation explicitly identifies itself as a modified,
experimental, or derived HingeZero variant, these equations define
**HingeZero Core**.
