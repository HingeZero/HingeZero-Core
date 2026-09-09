# HingeZero Core

HingeZero is a deterministic associative-memory mechanism built around a
**hinge-at-zero nonlinearity**.

The canonical HingeZero operator is:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

The canonical iterative recall update is:

```text
h_t = W @ x_t
φ_t = φ(h_t; α)
x_{t+1} = (1 - λ)x_t + εφ_t
```

The reference implementation is intentionally minimal.

---

## Overview

HingeZero applies a nonlinear transformation to the associative memory field
before updating the current state.

The memory field is:

```text
h = W @ x
```

The HingeZero nonlinear response is:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

The state update is:

```text
x_next = (1 - λ)x + εφ
```

Combined:

```text
h_t     = W @ x_t
φ_t     = tanh(h_t) + α·tanh(2h_t)
x_{t+1} = (1 - λ)x_t + εφ_t
```

Equivalent compact form:

```text
x_{t+1} = (1 - λ)x_t + ε·φ(Wx_t; α)
```

---

## Canonical HingeZero Core

The following implementation defines the reference HingeZero Core used by this
repository:

```python
def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha*np.tanh(2*h)

def hz_recall(W, x0, steps=2,
              alpha=0.25,
              eps=0.10,
              lam=0.02):

    x = x0.copy()

    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1-lam)*x + eps*phi

    return x
```

This code should be treated as the canonical reference implementation.

---

## Mathematical Definition

### HingeZero nonlinear operator

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

where:

```text
h       local associative-memory field
α       weighting of the second nonlinear term
φ       HingeZero nonlinear response
```

### Associative field

```text
h_t = W @ x_t
```

where:

```text
W       associative-memory matrix
x_t     current state
h_t     resulting local field
```

### State update

```text
x_{t+1} = (1 - λ)x_t + εφ(h_t; α)
```

where:

```text
λ       state decay / retention coefficient
ε       HingeZero update magnitude
```

Therefore:

```text
x_{t+1} =
    (1 - λ)x_t
    + ε[tanh(Wx_t) + α·tanh(2Wx_t)]
```

---

## Default Parameters

The reference HingeZero Core uses:

```text
alpha = 0.25
eps   = 0.10
lam   = 0.02
steps = 2
```

or mathematically:

```text
α = 0.25
ε = 0.10
λ = 0.02
```

These are the canonical default values used by the core implementation.

They may be changed experimentally without changing the definition of the
HingeZero operator itself.

---

## Core Functions

### `hinge_phi`

```python
def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha*np.tanh(2*h)
```

Mathematically:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

This is the defining HingeZero nonlinear operator.

---

### `hz_recall`

```python
def hz_recall(W, x0, steps=2,
              alpha=0.25,
              eps=0.10,
              lam=0.02):

    x = x0.copy()

    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1-lam)*x + eps*phi

    return x
```

Each recall iteration performs three operations:

```text
1. Calculate associative field

   h = W @ x

2. Apply HingeZero

   phi = tanh(h) + alpha*tanh(2*h)

3. Update state

   x = (1-lam)*x + eps*phi
```

The process repeats for the requested number of steps.

---

## Key Properties

### Deterministic update

The HingeZero recall function contains no random operation.

For fixed:

```text
W
x0
alpha
eps
lam
steps
```

the same update sequence is produced, subject to normal numerical differences
between software, hardware, and floating-point implementations.

---

### Nonlinear response

The HingeZero operator combines:

```text
tanh(h)
```

with:

```text
α·tanh(2h)
```

giving:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

---

### Near-zero response

Both nonlinear terms are centred at zero.

The second term changes the response around the origin:

```text
α·tanh(2h)
```

while retaining a smooth bounded nonlinear structure.

This behaviour is the origin of the name:

```text
HingeZero
```

---

### Minimal implementation

The essential recall mechanism is deliberately small:

```python
x = x0.copy()

for _ in range(steps):
    h = W @ x
    phi = hinge_phi(h, alpha)
    x = (1-lam)*x + eps*phi
```

This makes the core easy to inspect, reproduce, port, and benchmark.

---

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `alpha` | `0.25` | Weight of the `tanh(2h)` term |
| `eps` | `0.10` | Magnitude of the nonlinear update |
| `lam` | `0.02` | State decay / retention coefficient |
| `steps` | `2` | Number of recall iterations |

---

## Alpha

`alpha` controls the contribution of:

```text
tanh(2h)
```

to the HingeZero response.

The operator is:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

At the default:

```text
α = 0.25
```

this becomes:

```text
φ(h) = tanh(h) + 0.25·tanh(2h)
```

---

## Epsilon

`eps` corresponds to:

```text
ε
```

and controls how strongly the HingeZero response contributes to the next state.

```text
x_{t+1} = (1 - λ)x_t + εφ_t
```

Default:

```text
ε = 0.10
```

---

## Lambda

`lam` corresponds to:

```text
λ
```

and appears in:

```text
(1 - λ)x_t
```

Default:

```text
λ = 0.02
```

---

## Steps

`steps` controls the number of repeated HingeZero recall updates.

Default:

```text
steps = 2
```

The core algorithm therefore performs:

```text
x_0
 ↓
HZ update
 ↓
x_1
 ↓
HZ update
 ↓
x_2
```

with the default configuration.

---

## Memory Matrix

HingeZero Core does not require one specific method for constructing `W`.

The memory architecture supplies:

```text
W
```

and HingeZero operates on the resulting field:

```text
h = W @ x
```

This deliberately separates:

```text
memory construction
```

from:

```text
HingeZero recall dynamics
```

Conceptually:

```text
stored information
       ↓
memory matrix W
       ↓
query x
       ↓
h = W @ x
       ↓
HingeZero φ(h; α)
       ↓
updated state
```

---

## Basic Usage

```python
import numpy as np


def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha*np.tanh(2*h)


def hz_recall(W, x0, steps=2,
              alpha=0.25,
              eps=0.10,
              lam=0.02):

    x = x0.copy()

    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1-lam)*x + eps*phi

    return x
```

Assuming an associative-memory matrix `W` and an initial query `x0` already
exist:

```python
recalled = hz_recall(
    W,
    x0,
    steps=2,
    alpha=0.25,
    eps=0.10,
    lam=0.02
)
```

---

## Notation Standard

For consistency across HingeZero code, documentation, experiments, and
publications, the following notation should be used:

```text
x       current state
x0      initial query
x_t     state at iteration t

W       associative-memory matrix

h       associative-memory field
h_t     associative-memory field at iteration t

phi     Python variable containing the HingeZero response
φ       mathematical HingeZero operator

alpha   Python parameter
α       mathematical alpha parameter

eps     Python parameter
ε       mathematical epsilon parameter

lam     Python parameter
λ       mathematical lambda parameter

steps   number of recall iterations
```

---

## Canonical Mapping

The Python code:

```python
h = W @ x
```

corresponds to:

```text
h_t = W x_t
```

The Python code:

```python
phi = hinge_phi(h, alpha)
```

corresponds to:

```text
φ(h_t; α) = tanh(h_t) + α·tanh(2h_t)
```

The Python code:

```python
x = (1-lam)*x + eps*phi
```

corresponds to:

```text
x_{t+1} = (1 - λ)x_t + εφ(h_t; α)
```

---

## Determinism

The HingeZero recall mechanism is deterministic.

The following function contains no stochastic operation:

```python
def hz_recall(W, x0, steps=2,
              alpha=0.25,
              eps=0.10,
              lam=0.02):

    x = x0.copy()

    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1-lam)*x + eps*phi

    return x
```

Randomness may still be used externally when:

```text
generating test data
constructing experimental memories
adding corruption
adding noise
sampling benchmark queries
```

That randomness belongs to the experiment, not to HingeZero recall itself.

---

## Quantisation

The core definition is independent of a particular storage precision.

The reference Python version uses NumPy floating-point operations.

Experimental implementations can investigate:

```text
FP32
FP16
INT8
binary storage
1-bit packed representations
```

while retaining the same conceptual HingeZero operator or an explicitly
documented approximation of it.

Quantised variants should clearly state where their implementation differs from
the canonical floating-point reference.

---

## Repository Scope

This repository contains the **HingeZero Core**.

Its purpose is to establish:

```text
the canonical operator
the canonical recurrence
the default parameters
the minimal implementation
the notation standard
```

Application-specific systems, wrappers, tracking systems, retrieval engines,
quantised implementations, and experimental variants should be documented
separately.

---

## Canonical Definition

Unless explicitly labelled as an experimental or derived variant, HingeZero
should refer to:

```text
φ(h; α) = tanh(h) + α·tanh(2h)
```

with:

```text
h_t = W @ x_t
```

and:

```text
x_{t+1} = (1 - λ)x_t + εφ(h_t; α)
```

using the reference defaults:

```text
α = 0.25
ε = 0.10
λ = 0.02
steps = 2
```

---

## What's Included

This repository contains only the HingeZero core:

- Core implementation
- Mathematical definition
- Canonical notation
- Parameter definitions
- Recall dynamics
- Implementation guidance

Experimental extensions should be maintained separately so that the reference
core remains stable.

---

## Related Implementations

### HingeZero.py

Full Python implementation:

https://github.com/HingeZero/HingeZero.py

### HingeZero-1Bit

1-bit / packed-memory implementation:

https://github.com/HingeZero/HingeZero-1Bit

---

## Citation

If you use HingeZero in research, please cite:

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

- `CORE_IMPLEMENTATION.md` — implementation and mathematical guidance
- `HingeZero.py` — extended Python implementation
- `HingeZero-1Bit` — specialised low-bit implementation
