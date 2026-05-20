# HingeZero Core Implementation Guide

## Mathematical Foundation

### The Hinge-at-Zero Nonlinearity

The core nonlinearity is defined as:

```
return np.tanh(h) + alpha*np.tanh(2*h)
```

Where:
- `y` is the pre-activation input
- `α` is the balance parameter (typically 0.25)

**Key properties:**
- Smooth and differentiable everywhere
- Symmetric around zero
- Preserves near-zero values instead of forcing collapse
- Creates an active equilibrium region near zero
- Well-behaved for quantisation

### Update Dynamics

The recall process uses a weighted combination of the current state and the nonlinearity output:

```
x_{t+1} = (1 - λ)·x_t + ε·φ(h_t)
```

Where:
- `x_t` is the state at time t
- `h_t = W·x_t` is the preactivation (memory matrix applied to state)
- `φ` is the hinge-at-zero nonlinearity
- `ε` is the step size (learning/update rate)
- `λ` is the decay factor (controls state retention)

## Implementation Details

### Core Function: `hinge_phi`

```python
def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha*np.tanh(2*h)
```

**Usage:**
```python
import numpy as np

# Single value
output = hinge_phi(0.5)  # Returns approximately 0.52

# Vector
h = np.array([0.0, 0.5, -0.5])
output = hinge_phi(h)  # Element-wise application
```

### Core Function: `hz_recall`

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

**Usage:**
```python
# Create a memory matrix (typically learned or designed)
W = np.random.randn(100, 100) * 0.1

# Define an initial state (noisy query)
x0 = np.random.randn(100) * 0.5

# Perform recall
recalled = hz_recall(W, x0, steps=5, alpha=0.25, eps=0.10, lam=0.02)
```

## Parameter Tuning

### `alpha` (Nonlinearity Balance)
- **Default:** 0.25
- **Range:** 0.0 to ~1.0
- **Effect:** Higher values increase the influence of the tanh(2y) term
- **Tuning:** Start with default; increase for stronger nonlinearity, decrease for smoother dynamics

### `eps` (Step Size)
- **Default:** 0.10
- **Range:** 0.01 to 0.5 (typically)
- **Effect:** Controls the magnitude of updates from the nonlinearity
- **Tuning:** Smaller values = slower, more stable convergence; larger values = faster but risking oscillation

### `lam` (Decay Factor)
- **Default:** 0.02
- **Range:** 0.0 to 0.5 (typically)
- **Effect:** Fraction of state that is "forgotten" each step
- **Tuning:** Higher values accelerate convergence; lower values preserve history longer

### `steps` (Recall Iterations)
- **Default:** 2
- **Range:** 1 to 10+ (depends on problem)
- **Effect:** More iterations allow deeper pattern refinement
- **Tuning:** Increase until convergence or until retrieval quality plateaus

## Stability Considerations

1. **Weight matrix initialization:** Keep `|W|` entries small (e.g., ~0.1 scale) to avoid saturation
2. **State norms:** Monitor that state values don't explode; adjust `eps` and `lam` if needed
3. **Convergence:** Typically converges within 2-5 steps; excessive iterations may indicate parameter mismatch

## Quantisation

HingeZero is quantisation-friendly:
- `tanh` can be efficiently approximated with lookup tables
- The smooth nonlinearity tolerates low-bit representations
- Integer fixed-point arithmetic is feasible with proper scaling

## Related Repositories

- **HingeZero.py** – Full Python implementation with utilities
- **HingeZero-1Bit** – Packed 1-bit memory variant
- **HingeZero-** – Main research repository

## License

GNU General Public License v3.0
