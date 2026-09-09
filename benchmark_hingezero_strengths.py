import numpy as np
from time import perf_counter


# ============================================================
# HINGEZERO CANONICAL CORE
# ============================================================

def hinge_phi(h, alpha=0.25):
    return np.tanh(h) + alpha * np.tanh(2 * h)


def hz_recall(W, x0, steps=2,
              alpha=0.25,
              eps=0.10,
              lam=0.02):

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
# BENCHMARK CONFIG
# ============================================================

SEED = 2026

DIMENSIONS = [64, 128, 256]

MEMORY_COUNTS = [8, 32, 128]

CORRUPTIONS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

TRIALS = 20

ALPHA = 0.25
EPS = 0.10
LAM = 0.02
STEPS = 2


# ============================================================
# HELPERS
# ============================================================

def build_memory_matrix(memories):

    n = memories.shape[1]

    W = memories.T @ memories / n

    np.fill_diagonal(W, 0.0)

    return W


def corrupt_pattern(pattern, corruption, rng):

    x = pattern.copy()

    n = len(x)

    flips = int(round(n * corruption))

    idx = rng.choice(
        n,
        size=flips,
        replace=False
    )

    x[idx] *= -1.0

    return x


def all_cosines(query, memories):

    q_norm = np.linalg.norm(query)

    if q_norm == 0:
        return np.zeros(len(memories))

    m_norms = np.linalg.norm(
        memories,
        axis=1
    )

    denom = m_norms * q_norm

    scores = (
        memories @ query
    ) / denom

    return scores


# ============================================================
# SINGLE BENCHMARK CONDITION
# ============================================================

def run_condition(
    N,
    P,
    corruption,
    trials,
    rng
):

    memories = rng.choice(
        [-1.0, 1.0],
        size=(P, N)
    )

    W = build_memory_matrix(memories)

    baseline_correct = 0
    hz_correct = 0

    baseline_target_cos = []
    hz_target_cos = []

    baseline_margin = []
    hz_margin = []

    hz_improved_cos = 0
    hz_improved_margin = 0

    deterministic_ok = True

    for _ in range(trials):

        target_id = int(
            rng.integers(0, P)
        )

        target = memories[
            target_id
        ].copy()

        query = corrupt_pattern(
            target,
            corruption,
            rng
        )

        # ----------------------------------------------------
        # Baseline nearest-memory retrieval
        # ----------------------------------------------------

        baseline_scores = all_cosines(
            query,
            memories
        )

        baseline_id = int(
            np.argmax(
                baseline_scores
            )
        )

        b_target = float(
            baseline_scores[
                target_id
            ]
        )

        b_other = np.delete(
            baseline_scores,
            target_id
        )

        b_best_distractor = float(
            np.max(
                b_other
            )
        )

        b_margin = (
            b_target
            - b_best_distractor
        )

        # ----------------------------------------------------
        # HingeZero refinement
        # ----------------------------------------------------

        recalled = hz_recall(
            W,
            query,
            steps=STEPS,
            alpha=ALPHA,
            eps=EPS,
            lam=LAM
        )

        repeat = hz_recall(
            W,
            query,
            steps=STEPS,
            alpha=ALPHA,
            eps=EPS,
            lam=LAM
        )

        if not np.array_equal(
            recalled,
            repeat
        ):
            deterministic_ok = False

        hz_scores = all_cosines(
            recalled,
            memories
        )

        hz_id = int(
            np.argmax(
                hz_scores
            )
        )

        h_target = float(
            hz_scores[
                target_id
            ]
        )

        h_other = np.delete(
            hz_scores,
            target_id
        )

        h_best_distractor = float(
            np.max(
                h_other
            )
        )

        h_margin = (
            h_target
            - h_best_distractor
        )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        baseline_correct += int(
            baseline_id
            == target_id
        )

        hz_correct += int(
            hz_id
            == target_id
        )

        baseline_target_cos.append(
            b_target
        )

        hz_target_cos.append(
            h_target
        )

        baseline_margin.append(
            b_margin
        )

        hz_margin.append(
            h_margin
        )

        hz_improved_cos += int(
            h_target
            > b_target
        )

        hz_improved_margin += int(
            h_margin
            > b_margin
        )

    result = {

        "N":
            N,

        "P":
            P,

        "corruption":
            corruption,

        "baseline_acc":
            baseline_correct
            / trials,

        "hz_acc":
            hz_correct
            / trials,

        "baseline_cos":
            float(
                np.mean(
                    baseline_target_cos
                )
            ),

        "hz_cos":
            float(
                np.mean(
                    hz_target_cos
                )
            ),

        "baseline_margin":
            float(
                np.mean(
                    baseline_margin
                )
            ),

        "hz_margin":
            float(
                np.mean(
                    hz_margin
                )
            ),

        "cos_improved_rate":
            hz_improved_cos
            / trials,

        "margin_improved_rate":
            hz_improved_margin
            / trials,

        "deterministic":
            deterministic_ok,
    }

    return result


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    rng = np.random.default_rng(
        SEED
    )

    print("=" * 90)
    print(
        "HINGEZERO STRENGTHS BENCHMARK"
    )
    print("=" * 90)

    print(
        f"seed        : {SEED}"
    )
    print(
        f"dimensions  : {DIMENSIONS}"
    )
    print(
        f"memories    : {MEMORY_COUNTS}"
    )
    print(
        f"corruptions : {CORRUPTIONS}"
    )
    print(
        f"trials      : {TRIALS}"
    )

    print()

    start = perf_counter()

    results = []

    for N in DIMENSIONS:

        for P in MEMORY_COUNTS:

            for corruption in CORRUPTIONS:

                result = run_condition(
                    N,
                    P,
                    corruption,
                    TRIALS,
                    rng
                )

                results.append(
                    result
                )

                delta_acc = (
                    result["hz_acc"]
                    - result["baseline_acc"]
                )

                delta_cos = (
                    result["hz_cos"]
                    - result["baseline_cos"]
                )

                delta_margin = (
                    result["hz_margin"]
                    - result["baseline_margin"]
                )

                print(
                    f"N={N:3d} "
                    f"P={P:3d} "
                    f"noise={corruption:4.0%} | "
                    f"ACC "
                    f"B={result['baseline_acc']:.3f} "
                    f"HZ={result['hz_acc']:.3f} "
                    f"Δ={delta_acc:+.3f} | "
                    f"COS "
                    f"B={result['baseline_cos']:.4f} "
                    f"HZ={result['hz_cos']:.4f} "
                    f"Δ={delta_cos:+.4f} | "
                    f"MARGIN "
                    f"B={result['baseline_margin']:+.4f} "
                    f"HZ={result['hz_margin']:+.4f} "
                    f"Δ={delta_margin:+.4f}"
                )

    elapsed = (
        perf_counter()
        - start
    )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    total_conditions = len(results)

    acc_better = sum(
        r["hz_acc"]
        > r["baseline_acc"]
        for r in results
    )

    acc_equal = sum(
        r["hz_acc"]
        == r["baseline_acc"]
        for r in results
    )

    cos_better = sum(
        r["hz_cos"]
        > r["baseline_cos"]
        for r in results
    )

    margin_better = sum(
        r["hz_margin"]
        > r["baseline_margin"]
        for r in results
    )

    all_deterministic = all(
        r["deterministic"]
        for r in results
    )

    mean_acc_delta = float(
        np.mean([
            r["hz_acc"]
            - r["baseline_acc"]
            for r in results
        ])
    )

    mean_cos_delta = float(
        np.mean([
            r["hz_cos"]
            - r["baseline_cos"]
            for r in results
        ])
    )

    mean_margin_delta = float(
        np.mean([
            r["hz_margin"]
            - r["baseline_margin"]
            for r in results
        ])
    )

    print(
        f"conditions tested       : {total_conditions}"
    )

    print(
        f"HZ accuracy better      : {acc_better}/{total_conditions}"
    )

    print(
        f"HZ accuracy equal       : {acc_equal}/{total_conditions}"
    )

    print(
        f"HZ cosine better        : {cos_better}/{total_conditions}"
    )

    print(
        f"HZ margin better        : {margin_better}/{total_conditions}"
    )

    print(
        f"mean accuracy delta     : {mean_acc_delta:+.4f}"
    )

    print(
        f"mean cosine delta       : {mean_cos_delta:+.4f}"
    )

    print(
        f"mean margin delta       : {mean_margin_delta:+.4f}"
    )

    print(
        f"deterministic all cases : {all_deterministic}"
    )

    print(
        f"runtime                 : {elapsed:.2f} s"
    )

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    print(
        "This benchmark compares nearest-memory retrieval "
        "before and after canonical HingeZero refinement."
    )

    print(
        "A positive delta means HingeZero improved that metric."
    )

    print(
        "Accuracy, target cosine, and target-vs-distractor margin "
        "are reported separately so improvements cannot be hidden "
        "inside a single score."
    )

    print()
    print(
        "HINGEZERO STRENGTHS BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
