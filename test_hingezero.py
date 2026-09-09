import numpy as np


# ============================================================
# HINGEZERO CANONICAL CORE
# ============================================================

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


# ============================================================
# SMALL DETERMINISTIC TEST
# ============================================================

def main():

    print("=" * 60)
    print("HINGEZERO CORE — DETERMINISTIC SMOKE TEST")
    print("=" * 60)

    W = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    x0 = np.array([
        1.0,
        -1.0
    ])

    print("\nInput:")
    print(x0)

    out1 = hz_recall(W, x0)
    out2 = hz_recall(W, x0)

    print("\nOutput:")
    print(out1)

    # --------------------------------------------------------
    # Basic validity
    # --------------------------------------------------------

    assert out1.shape == x0.shape, \
        "Output shape changed."

    assert np.all(np.isfinite(out1)), \
        "Non-finite values detected."

    # --------------------------------------------------------
    # Determinism
    # --------------------------------------------------------

    assert np.array_equal(out1, out2), \
        "Repeated recall produced different results."

    # --------------------------------------------------------
    # Canonical operator verification
    # --------------------------------------------------------

    h = W @ x0

    expected_phi = (
        np.tanh(h)
        + 0.25*np.tanh(2*h)
    )

    actual_phi = hinge_phi(h, 0.25)

    assert np.allclose(
        actual_phi,
        expected_phi
    ), "Canonical HingeZero operator mismatch."

    print("\nDeterministic repeat:")
    print(out2)

    print("\n✓ finite output")
    print("✓ shape preserved")
    print("✓ deterministic repeat")
    print("✓ canonical φ(h) verified")

    print("\nHINGEZERO CORE TEST PASSED")


if __name__ == "__main__":
    main()
