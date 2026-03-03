// C++ implementation of the GreCov BFS algorithm
//
// Best-first enumeration of multinomial count vectors in decreasing
// probability order.  Returns two-sided tail probabilities and the
// weighted sums sum_x P(x)*x for each tail, needed for analytic gradients.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

// ─── Hash for state vectors ────────────────────────────────────────

struct StateHash {
    std::size_t operator()(const std::vector<int32_t>& v) const noexcept {
        // FNV-1a
        std::size_t h = 14695981039346656037ULL;
        for (auto x : v) {
            h ^= static_cast<std::size_t>(static_cast<uint32_t>(x));
            h *= 1099511628211ULL;
        }
        return h;
    }
};

// ─── Balanced rounding (largest-remainder method) ──────────────────

static std::vector<int32_t> start_counts(const std::vector<double>& p, int n) {
    int d = static_cast<int>(p.size());
    std::vector<int32_t> counts(d);
    std::vector<double> frac(d);

    int total = 0;
    for (int i = 0; i < d; ++i) {
        double x = p[i] * n;
        counts[i] = static_cast<int32_t>(std::floor(x));
        frac[i] = x - counts[i];
        total += counts[i];
    }

    int remainder = n - total;

    // Indices sorted by descending fractional part
    std::vector<int> idx(d);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&frac](int a, int b) { return frac[a] > frac[b]; });

    for (int r = 0; r < remainder; ++r) {
        counts[idx[r]] += 1;
    }
    return counts;
}

// ─── Precompute log-factorials ─────────────────────────────────────

static std::vector<double> log_factorials(int n) {
    std::vector<double> lf(n + 1, 0.0);
    for (int i = 1; i <= n; ++i) {
        lf[i] = lf[i - 1] + std::log(static_cast<double>(i));
    }
    return lf;
}

// ─── BFS result ────────────────────────────────────────────────────

struct BFSResult {
    double prob_left;
    double prob_right;
    std::vector<double> wsum_left;
    std::vector<double> wsum_right;
    double explored_mass;
    int64_t states_explored;
};

// ─── GreCov BFS ────────────────────────────────────────────────────

static BFSResult grecov_bfs_impl(
    const std::vector<double>& p_raw,
    const std::vector<double>& v,
    double S_obs,
    int n,
    double eps)
{
    int d = static_cast<int>(p_raw.size());

    // Stabilise probabilities
    constexpr double MIN_P = 1e-300;
    std::vector<double> p(d);
    double p_sum = 0.0;
    for (int i = 0; i < d; ++i) {
        p[i] = std::max(p_raw[i], MIN_P);
        p_sum += p[i];
    }
    for (int i = 0; i < d; ++i) p[i] /= p_sum;

    double min_p = *std::min_element(p.begin(), p.end());
    double residual_tol = eps * min_p / std::max(1, n);

    std::vector<double> log_p(d);
    for (int i = 0; i < d; ++i) log_p[i] = std::log(p[i]);

    auto log_fact = log_factorials(n);

    // Compute log-probability of a state
    auto log_prob = [&](const std::vector<int32_t>& c) -> double {
        double val = log_fact[n];
        for (int i = 0; i < d; ++i) {
            val -= log_fact[c[i]];
            val += c[i] * log_p[i];
        }
        return val;
    };

    // Start from approximate mode
    auto start = start_counts(p, n);
    double start_lp = log_prob(start);

    // Max-heap by logP (negate for min-heap)
    using Entry = std::pair<double, std::vector<int32_t>>;
    auto cmp = [](const Entry& a, const Entry& b) { return a.first < b.first; };
    std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);

    std::unordered_set<std::vector<int32_t>, StateHash> visited;
    visited.reserve(1 << 16);

    heap.push({start_lp, start});
    visited.insert(start);

    double P_explored = 0.0;
    double P_left = 0.0, P_right = 0.0, P_equal = 0.0;
    std::vector<double> wsum_left(d, 0.0), wsum_right(d, 0.0), wsum_equal(d, 0.0);
    int64_t states_explored = 0;

    while (!heap.empty()) {
        auto [logP, counts] = std::move(const_cast<Entry&>(heap.top()));
        heap.pop();

        ++states_explored;
        double P_state = std::exp(logP);
        P_explored += P_state;

        // Compute S = v^T x
        double s_val = 0.0;
        for (int i = 0; i < d; ++i) s_val += counts[i] * v[i];

        if (s_val < S_obs) {
            P_left += P_state;
            for (int i = 0; i < d; ++i) wsum_left[i] += P_state * counts[i];
        } else if (s_val > S_obs) {
            P_right += P_state;
            for (int i = 0; i < d; ++i) wsum_right[i] += P_state * counts[i];
        } else {
            P_equal += P_state;
            for (int i = 0; i < d; ++i) wsum_equal[i] += P_state * counts[i];
        }

        if (1.0 - P_explored <= residual_tol) break;

        // Generate neighbours: transfer one unit from j to i
        for (int j = 0; j < d; ++j) {
            if (counts[j] == 0) continue;
            for (int i = 0; i < d; ++i) {
                if (i == j) continue;

                auto neighbor = counts;
                neighbor[i] += 1;
                neighbor[j] -= 1;

                if (!visited.insert(neighbor).second) continue;

                double logP_n = logP
                    + std::log(static_cast<double>(counts[j]))
                    - std::log(static_cast<double>(counts[i] + 1))
                    + log_p[i] - log_p[j];

                heap.push({logP_n, std::move(neighbor)});
            }
        }
    }

    // Merge equal into both tails
    BFSResult result;
    result.prob_left = P_left + P_equal;
    result.prob_right = P_right + P_equal;
    result.wsum_left.resize(d);
    result.wsum_right.resize(d);
    for (int i = 0; i < d; ++i) {
        result.wsum_left[i] = wsum_left[i] + wsum_equal[i];
        result.wsum_right[i] = wsum_right[i] + wsum_equal[i];
    }
    result.explored_mass = std::min(P_explored, 1.0);
    result.states_explored = states_explored;
    return result;
}

// ─── nanobind module ───────────────────────────────────────────────

NB_MODULE(_ext, m) {
    m.doc() = "C++ for the GreCov algorithm";

    m.def("grecov_bfs",
        [](const std::vector<double>& p,
           const std::vector<double>& v,
           double S_obs, int n, double eps) -> nb::dict {

            auto res = grecov_bfs_impl(p, v, S_obs, n, eps);

            nb::dict d;
            d["prob_left"] = res.prob_left;
            d["prob_right"] = res.prob_right;
            d["wsum_left"] = res.wsum_left;
            d["wsum_right"] = res.wsum_right;
            d["explored_mass"] = res.explored_mass;
            d["states_explored"] = res.states_explored;
            return d;
        },
        nb::arg("p"), nb::arg("v"), nb::arg("S_obs"), nb::arg("n"), nb::arg("eps"),
        "Run the GreCov best-first search algorithm.\n\n"
        "Parameters\n"
        "----------\n"
        "p : list of float — probability vector\n"
        "v : list of float — value vector\n"
        "S_obs : float — observed test statistic v^T x\n"
        "n : int — total count\n"
        "eps : float — stopping tolerance\n\n"
        "Returns\n"
        "-------\n"
        "dict with keys: prob_left, prob_right, wsum_left, wsum_right,\n"
        "explored_mass, states_explored"
    );
}
