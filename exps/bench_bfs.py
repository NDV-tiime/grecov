"""Timing benchmarks for the C++ BFS hot path.

Uses p vectors extracted from a real solver run (confidence_interval with
counts=[30,25,20,15,10], values=[0,1,2,3,4], alpha=0.05).
"""

import argparse
import timeit

from grecov._ext import grecov_bfs, grecov_mass_bfs

# Shared parameters
v = [0, 1, 2, 3, 4]
eps = 5e-5  # eps_ratio=1e-3 * alpha=0.05
x_obs = [30, 25, 20, 15, 10]

# Probability vectors from solver extremes and common points
p_lower = [
    7.14093467e-01,
    1.36442907e-12,
    1.20061383e-12,
    9.95174051e-03,
    2.75954792e-01,
]
p_upper = [
    5.20119344e-01,
    1.05923135e-12,
    1.14109740e-02,
    1.34394004e-12,
    4.68469682e-01,
]
p_mle = [0.3, 0.25, 0.2, 0.15, 0.1]
p_uniform = [0.2, 0.2, 0.2, 0.2, 0.2]

cases = {
    "p_lower": p_lower,
    "p_upper": p_upper,
    "p_mle": p_mle,
    "p_uniform": p_uniform,
}


def _fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:>8.1f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:>8.1f} ms"
    else:
        return f"{seconds:>8.2f}  s"


def _auto_bench(fn, target_time=2.0):
    """Run fn enough times to get a stable measurement."""
    # Warm up
    fn()
    # Calibrate
    t = timeit.timeit(fn, number=1)
    n = max(1, int(target_time / t))
    total = timeit.timeit(fn, number=n)
    return total / n


def bench_tail_bfs():
    for n in [20, 50, 100]:
        s_obs = sum(ci * vi for ci, vi in zip(x_obs, v)) * n / 100
        print(f"\n=== grecov_bfs (tail) | n={n}, k=5, eps={eps} ===")
        print(f"{'case':<12} {'time/call':>12} {'states':>10}")
        print("-" * 36)
        for name, p in cases.items():
            res = grecov_bfs(p, v, s_obs, n, eps)
            t = _auto_bench(lambda p=p: grecov_bfs(p, v, s_obs, n, eps))
            print(f"{name:<12} {_fmt_time(t)} {res['states_explored']:>10}")


def bench_mass_bfs():
    for n_scale, x_obs_scaled in [(1, x_obs), (2, [c * 2 for c in x_obs])]:
        n = sum(x_obs_scaled)
        print(f"\n=== grecov_mass_bfs (mass) | n={n}, k=5, eps={eps} ===")
        print(f"{'case':<12} {'time/call':>12} {'states':>10}")
        print("-" * 36)
        for name, p in cases.items():
            res = grecov_mass_bfs(p, x_obs_scaled, eps, 1e-8)
            t = _auto_bench(
                lambda p=p, xo=x_obs_scaled: grecov_mass_bfs(p, xo, eps, 1e-8)
            )
            print(f"{name:<12} {_fmt_time(t)} {res['states_explored']:>10}")


def bench_solver():
    from grecov.solver import confidence_interval

    test_cases = [
        {
            "counts": [30, 25, 20, 15, 10],
            "values": [0, 1, 2, 3, 4],
            "label": "k=5 n=100",
        },
        {"counts": [15, 12, 10, 8, 5], "values": [0, 1, 2, 3, 4], "label": "k=5 n=50"},
        {"counts": [8, 6, 4, 2], "values": [0, 1, 2, 3], "label": "k=4 n=20"},
    ]

    print("\n=== Full solver (confidence_interval, equal_tail) ===")
    print(f"{'case':<16} {'time':>12} {'bfs_calls':>10} {'bfs_states':>12}")
    print("-" * 52)
    for tc in test_cases:
        res = confidence_interval(tc["counts"], tc["values"], alpha=0.05)
        t = _auto_bench(
            lambda tc=tc: confidence_interval(tc["counts"], tc["values"], alpha=0.05)
        )
        print(
            f"{tc['label']:<16} {_fmt_time(t)} {res['bfs_calls']:>10} {res['bfs_total_states']:>12}"
        )
        print(f"  CI: [{res['lower']:.6f}, {res['upper']:.6f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfs", action="store_true", help="Run BFS microbenchmarks")
    parser.add_argument(
        "--solver", action="store_true", help="Run full solver benchmark"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all or (not args.bfs and not args.solver):
        args.bfs = args.solver = True

    if args.bfs:
        bench_tail_bfs()
        bench_mass_bfs()
    if args.solver:
        bench_solver()
