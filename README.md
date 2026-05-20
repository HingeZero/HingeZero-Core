# HingeZero Core

A deterministic associative memory mechanism built around a **hinge-at-zero nonlinearity**.

## Overview

HingeZero treats the neighbourhood around zero as an **active equilibrium region** rather than a passive dead zone. This allows opposing states to balance, hold structure, and resolve instead of collapsing into forced binary outputs.

**Mathematical basis:**
```
h(y; α) = tanh(y) + α·tanh(2y)
```

Core implementation is intentionally minimal (~9 executable lines).

## Key Features

✅ **Deterministic updates** – No randomness, reproducible results  
✅ **Noise tolerant** – Stable recall dynamics under perturbation  
✅ **Quantisation friendly** – Works with low-bit arithmetic  
✅ **Minimal footprint** – ~9 lines of core logic  
✅ **Reproducible** – No hidden state or initialization magic  
✅ **Easy to integrate** – Pure numpy, no dependencies  

## What's Included

This repository contains **only the HingeZero core**:
- Core implementation (Python)
- Mathematical documentation
- Parameter tuning guide
- No tests, wrappers, or hybrid variants

For full implementations, see:
- **[HingeZero.py](https://github.com/HingeZero/HingeZero.py)** – Complete Python package
- **[HingeZero-1Bit](https://github.com/HingeZero/HingeZero-1Bit)** – 1-bit packed variant

## Quick Start

### Core Functions

```python
import numpy as np

def hinge_phi(h, alpha=0.25):
    """Hinge-at-zero nonlinearity"""
    return np.tanh(h) + alpha*np.tanh(2*h)

def hz_recall(W, x0, steps=2, alpha=0.25, eps=0.10, lam=0.02):
    """Perform associative recall"""
    x = x0.copy()
    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1-lam)*x + eps*phi
    return x
```

### Basic Usage

```python
# Create or load a memory matrix
W = np.random.randn(100, 100) * 0.1

# Define a noisy query
x0 = np.random.randn(100) * 0.5

# Perform recall
recalled = hz_recall(W, x0, steps=5, alpha=0.25, eps=0.10, lam=0.02)
```

## Parameters Explained

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `alpha` | 0.25 | Nonlinearity balance (tanh weight) |
| `eps` | 0.10 | Step size / update magnitude |
| `lam` | 0.02 | State decay / history retention |
| `steps` | 2 | Number of recall iterations |

For detailed tuning guidance, see [CORE_IMPLEMENTATION.md](CORE_IMPLEMENTATION.md).

## Design Philosophy

- **Minimal:** Core logic is short, auditable, and transparent
- **Deterministic:** No randomness in recall dynamics
- **Stable:** Preserves near-zero structure instead of forcing collapse
- **Practical:** Designed for real-world integration and quantisation

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

## License

GNU General Public License v3.0 – See [LICENSE](LICENSE)

## Documentation

- **[CORE_IMPLEMENTATION.md](CORE_IMPLEMENTATION.md)** – Complete implementation guide with math, code examples, and tuning strategies
- **[HingeZero.py Repository](https://github.com/HingeZero/HingeZero.py)** – Full-featured Python implementation
