import numpy as np


# ============================================================
# HINGEZERO CANONICAL CORE
# ============================================================

def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha * np.tanh(2 * h)


def hz_recall(W, x0, steps=2, alpha=0.25, eps=0.10, lam=0.02):
    x = x0.astype(float).copy()

    for _ in range(steps):
        h = W @ x
        phi = hinge_phi(h, alpha)
        x = (1 - lam) * x + eps * phi

    return x


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# ============================================================
# SMALL ASSOCIATIVE MEMORY TEST
# ============================================================

def main():

    print("=" * 60)
    print("HINGEZERO — SMALL ASSOCIATIVE RECALL TEST")
    print("=" * 60)

    # Fixed seed makes the experiment reproducible.
    rng = np.random.default_rng(2026)

    N = 64
    P = 8

    alpha = 0.25
    eps = 0.10
    lam = 0.02
    steps = 2

    # --------------------------------------------------------
    # Create deterministic bipolar memories.
    # --------------------------------------------------------

    memories = rng.choice(
        [-1.0, 1.0],
        size=(P, N)
    )

    # --------------------------------------------------------
    # Hebbian associative memory matrix.
    # W = (1/N) sum(mu mu^T)
    # --------------------------------------------------------

    W = memories.T @ memories / N

    # Remove self-coupling.
    np.fill_diagonal(W, 0.0)

    # --------------------------------------------------------
    # Select one stored memory.
    # --------------------------------------------------------

    target_id = 3
    target = memories[target_id].copy()

    # --------------------------------------------------------
    # Corrupt exactly 20% of its coordinates.
    # --------------------------------------------------------

    corruption = 0.20
    flips = int(round(N * corruption))

    flip_idx = rng.choice(
        N,
        size=flips,
        replace=False
    )

    query = target.copy()
    query[flip_idx] *= -1

    # --------------------------------------------------------
    # HingeZero refinement.
    # --------------------------------------------------------

    recalled = hz_recall(
        W,
        query,
        steps=steps,
        alpha=alpha,
        eps=eps,
        lam=lam
    )

    # --------------------------------------------------------
    # Compare recalled state against every stored memory.
    # --------------------------------------------------------

    before_scores = np.array([
        cosine(query, m)
        for m in memories
    ])

    after_scores = np.array([
        cosine(recalled, m)
        for m in memories
    ])

    before_id = int(np.argmax(before_scores))
    recalled_id = int(np.argmax(after_scores))

    before_target = float(before_scores[target_id])
    after_target = float(after_scores[target_id])

    distractors = np.delete(after_scores, target_id)
    best_distractor = float(np.max(distractors))

    margin = after_target - best_distractor

    # --------------------------------------------------------
    # Determinism check.
    # --------------------------------------------------------

    repeat = hz_recall(
        W,
        query,
        steps=steps,
        alpha=alpha,
        eps=eps,
        lam=lam
    )

    deterministic = np.array_equal(recalled, repeat)

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    print(f"\nDimensions       : {N}")
    print(f"Stored memories  : {P}")
    print(f"Target ID        : {target_id}")
    print(f"Corruption       : {corruption:.0%}")
    print(f"Flipped bits     : {flips}")

    print("\nBefore HingeZero")
    print(f"Best memory      : {before_id}")
    print(f"Target cosine    : {before_target:.6f}")

    print("\nAfter HingeZero")
    print(f"Best memory      : {recalled_id}")
    print(f"Target cosine    : {after_target:.6f}")
    print(f"Best distractor  : {best_distractor:.6f}")
    print(f"Target margin    : {margin:+.6f}")

    print("\nVerification")
    print(f"Deterministic    : {deterministic}")

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert np.all(np.isfinite(recalled)), \
        "HingeZero produced non-finite values."

    assert recalled.shape == target.shape, \
        "HingeZero changed dimensionality."

    assert deterministic, \
        "Repeated HingeZero recall was not deterministic."

    assert recalled_id == target_id, \
        f"Recall failed: expected memory {target_id}, got {recalled_id}."

    assert margin > 0, \
        "Target did not beat the strongest distractor."

    print("\n✓ finite output")
    print("✓ shape preserved")
    print("✓ deterministic repeat")
    print("✓ correct target identified")
    print("✓ positive target margin")

    print("\nHINGEZERO ASSOCIATIVE RECALL TEST PASSED")


if __name__ == "__main__":
    main()
