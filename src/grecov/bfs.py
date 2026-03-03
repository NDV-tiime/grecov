"""
Python implementation of the GreCov BFS algorithm.

Best-first enumeration of multinomial count vectors in decreasing
probability order. Returns two-sided tail probabilities and the
conditional expectations needed for analytic gradients.
"""

import heapq
import math


def _start_counts(p, n):
    """Balanced rounding of n*p to integers."""
    k = len(p)
    floors = [int(math.floor(pi * n)) for pi in p]
    fracs = [pi * n - f for pi, f in zip(p, floors)]
    remainder = n - sum(floors)
    indices = sorted(range(k), key=lambda i: fracs[i], reverse=True)
    for r in range(remainder):
        floors[indices[r]] += 1
    return floors


def grecov_bfs(p, v, s_obs, n, eps=1e-6):
    """
    Parameters
    ----------
    p : list of float
        Probability vector.
    v : list of float
        Value vector for each category.
    s_obs : float
        Observed test statistic v^T x.
    n : int
        Total count (sum of observations).
    eps : float
        Stopping tolerance on unexplored mass.

    Returns
    -------
    dict with keys:
        prob_left, prob_right    — tail probabilities P(S <= s_obs), P(S >= s_obs)
        wsum_left, wsum_right — sum_x P(x)*x for each tail
        explored_mass            — total probability mass visited
        states_explored          — number of states popped from the heap
    """
    assert n > 0, "n must be positive"

    k = len(p)

    MIN_P = 1e-300
    p_stable = [max(pi, MIN_P) for pi in p]
    p_sum = sum(p_stable)
    p_stable = [pi / p_sum for pi in p_stable]

    min_p = min(p_stable)
    residual_tol = eps * min_p / max(1, n)

    log_p = [math.log(pi) for pi in p_stable]

    # Precompute log-factorials
    log_fact = [0.0] * (n + 1)
    for i in range(1, n + 1):
        log_fact[i] = log_fact[i - 1] + math.log(i)

    def log_prob(counts):
        val = log_fact[n]
        for ci, lp in zip(counts, log_p):
            val -= log_fact[ci]
            val += ci * lp
        return val

    # Start from the approximate mode
    start = tuple(_start_counts(p_stable, n))
    heap = [(-log_prob(start), start)]  # min-heap with negated keys = max-heap
    visited = {start}

    mass = 0.0
    prob_left = 0.0
    prob_right = 0.0
    prob_equal = 0.0
    wsum_left = [0.0] * k
    wsum_right = [0.0] * k
    wsum_equal = [0.0] * k
    states_explored = 0

    while heap:
        neg_lp, counts = heapq.heappop(heap)
        log_p_state = -neg_lp
        states_explored += 1

        p_state = math.exp(log_p_state)
        mass += p_state

        # Classify: s < s_obs (left), s > s_obs (right), s == s_obs (both)
        s_val = sum(ci * vi for ci, vi in zip(counts, v))

        if s_val < s_obs:
            prob_left += p_state
            for i in range(k):
                wsum_left[i] += p_state * counts[i]
        elif s_val > s_obs:
            prob_right += p_state
            for i in range(k):
                wsum_right[i] += p_state * counts[i]
        else:
            prob_equal += p_state
            for i in range(k):
                wsum_equal[i] += p_state * counts[i]

        if 1.0 - mass <= residual_tol:
            break

        # Generate neighbours: transfer one unit from j to i
        for j in range(k):
            if counts[j] == 0:
                continue
            for i in range(k):
                if i == j:
                    continue
                neighbor = list(counts)
                neighbor[i] += 1
                neighbor[j] -= 1
                neighbor = tuple(neighbor)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                log_p_neighbor = (log_p_state
                                  + math.log(counts[j])
                                  - math.log(counts[i] + 1)
                                  + log_p[i] - log_p[j])
                heapq.heappush(heap, (-log_p_neighbor, neighbor))

    # States with s == s_obs contribute to both tails
    return {
        "prob_left": prob_left + prob_equal,
        "prob_right": prob_right + prob_equal,
        "wsum_left": [l + e for l, e in zip(wsum_left, wsum_equal)],
        "wsum_right": [r + e for r, e in zip(wsum_right, wsum_equal)],
        "explored_mass": mass,
        "states_explored": states_explored,
    }
