// C++ implementation of the GreCov BFS algorithm — optimized version.
//
// Key optimizations over the naive implementation:
// - States are packed into uint64_t (for d<=8, counts<=255) to eliminate
//   all heap allocations in the hash set and priority queue.
// - Flat integer hashing instead of vector hashing.
// - Precomputed log(i) table to avoid repeated log() calls in neighbor gen.
// - Symmetric wsum2 accumulation with half the multiplies.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

static constexpr int MAX_D = 8;

// ─── Packed state: up to 8 categories, 8 bits each ──────────────────

// Pack d counts (each <= 255) into a uint64_t, big-endian by index
// so that numeric comparison of uint64_t matches lexicographic comparison.
inline uint64_t pack_state(const int32_t *c, int d) {
  uint64_t s = 0;
  for (int i = 0; i < d; ++i)
    s |= static_cast<uint64_t>(static_cast<uint8_t>(c[i])) << ((7 - i) * 8);
  return s;
}

inline void unpack_state(uint64_t s, int32_t *c, int d) {
  for (int i = 0; i < d; ++i)
    c[i] = static_cast<int32_t>((s >> ((7 - i) * 8)) & 0xFF);
}

// Get component i from packed state
inline int32_t packed_get(uint64_t s, int i) {
  return static_cast<int32_t>((s >> ((7 - i) * 8)) & 0xFF);
}

// Transfer one unit from j to i in packed state
inline uint64_t packed_transfer(uint64_t s, int i, int j) {
  constexpr uint64_t one = 1;
  return s + (one << ((7 - i) * 8)) - (one << ((7 - j) * 8));
}

// ─── Balanced rounding (largest-remainder method) ──────────────────

static void start_counts(const double *p, int n, int d, int32_t *out) {
  double frac[MAX_D];
  int total = 0;
  for (int i = 0; i < d; ++i) {
    double x = p[i] * n;
    out[i] = static_cast<int32_t>(std::floor(x));
    frac[i] = x - out[i];
    total += out[i];
  }
  int remainder = n - total;

  // Partial sort: find top `remainder` indices by descending fractional part
  int idx[MAX_D];
  for (int i = 0; i < d; ++i)
    idx[i] = i;
  std::partial_sort(idx, idx + remainder, idx + d,
                    [&frac](int a, int b) { return frac[a] > frac[b]; });
  for (int r = 0; r < remainder; ++r)
    out[idx[r]] += 1;
}

// ─── Precompute log-factorials ─────────────────────────────────────

static std::vector<double> log_factorials(int n) {
  std::vector<double> lf(n + 1, 0.0);
  for (int i = 1; i <= n; ++i)
    lf[i] = lf[i - 1] + std::log(static_cast<double>(i));
  return lf;
}

// ─── Precompute log(i) for i in [1..n] ─────────────────────────────

static std::vector<double> log_integers(int n) {
  std::vector<double> li(n + 2, 0.0);
  for (int i = 1; i <= n + 1; ++i)
    li[i] = std::log(static_cast<double>(i));
  return li;
}

// ─── Hash for uint64_t (splitmix64 finalizer) ──────────────────────

struct U64Hash {
  std::size_t operator()(uint64_t x) const noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<std::size_t>(x);
  }
};

// ─── Heap entry ─────────────────────────────────────────────────────

struct Entry {
  double logP;
  uint64_t state;

  bool operator<(const Entry &other) const {
    if (logP != other.logP)
      return logP < other.logP;
    return state >
           other.state; // smaller state = higher priority (for determinism)
  }
};

// ─── BFS result ────────────────────────────────────────────────────

struct BFSResult {
  double prob_left;
  double prob_right;
  std::vector<double> wsum_left;
  std::vector<double> wsum_right;
  std::vector<double> wsum2_left;
  std::vector<double> wsum2_right;
  double explored_mass;
  int64_t states_explored;
};

// ─── GreCov BFS (tail) ─────────────────────────────────────────────

static BFSResult grecov_bfs_impl(const std::vector<double> &p_raw,
                                 const std::vector<double> &v, double S_obs,
                                 int n, double eps) {
  int d = static_cast<int>(p_raw.size());

  // Stabilise probabilities
  constexpr double MIN_P = 1e-300;
  double p[MAX_D], log_p[MAX_D], v_arr[MAX_D];
  double p_sum = 0.0;
  for (int i = 0; i < d; ++i) {
    p[i] = std::max(p_raw[i], MIN_P);
    p_sum += p[i];
  }
  for (int i = 0; i < d; ++i) {
    p[i] /= p_sum;
    log_p[i] = std::log(p[i]);
    v_arr[i] = v[i];
  }

  auto log_fact = log_factorials(n);
  auto log_int = log_integers(n);

  // Compute log-probability of a packed state
  auto log_prob = [&](uint64_t s) -> double {
    double val = log_fact[n];
    for (int i = 0; i < d; ++i) {
      int32_t ci = packed_get(s, i);
      val -= log_fact[ci];
      val += ci * log_p[i];
    }
    return val;
  };

  // Start from approximate mode
  int32_t start_c[MAX_D];
  start_counts(p, n, d, start_c);
  uint64_t start_s = pack_state(start_c, d);
  double start_lp = log_prob(start_s);

  std::priority_queue<Entry> heap;
  std::unordered_set<uint64_t, U64Hash> visited;
  visited.reserve(1 << 16);

  heap.push({start_lp, start_s});
  visited.insert(start_s);

  double P_explored = 0.0;
  int dk = d * d;

  // Per-tail accumulators: 0=left, 1=right, 2=equal
  double probs[3] = {};
  double wsums[3][MAX_D] = {};
  // wsum2 needs d*d, use heap for that
  std::vector<double> wsum2s_0(dk, 0.0), wsum2s_1(dk, 0.0), wsum2s_2(dk, 0.0);
  double *wsum2_ptrs[3] = {wsum2s_0.data(), wsum2s_1.data(), wsum2s_2.data()};

  int64_t states_explored = 0;

  while (!heap.empty()) {
    auto top = heap.top();
    heap.pop();
    double logP = top.logP;
    uint64_t state = top.state;

    ++states_explored;
    double P_state = std::exp(logP);
    P_explored += P_state;

    // Unpack counts for accumulation
    int32_t c[MAX_D];
    unpack_state(state, c, d);

    // Compute S = v^T x
    double s_val = 0.0;
    for (int i = 0; i < d; ++i)
      s_val += c[i] * v_arr[i];

    // Classify and accumulate
    int side;
    if (s_val < S_obs)
      side = 0;
    else if (s_val > S_obs)
      side = 1;
    else
      side = 2;

    probs[side] += P_state;
    double *ws = wsums[side];
    double *ws2 = wsum2_ptrs[side];
    for (int i = 0; i < d; ++i) {
      double pci = P_state * c[i];
      ws[i] += pci;
      // Diagonal
      ws2[i * d + i] += pci * c[i];
      // Off-diagonal (symmetric)
      for (int j = i + 1; j < d; ++j) {
        double val = pci * c[j];
        ws2[i * d + j] += val;
        ws2[j * d + i] += val;
      }
    }

    if (1.0 - eps <= P_explored)
      break;

    // Generate neighbours: transfer one unit from j to i
    for (int j = 0; j < d; ++j) {
      if (c[j] == 0)
        continue;
      for (int i = 0; i < d; ++i) {
        if (i == j)
          continue;

        uint64_t neighbor = packed_transfer(state, i, j);
        if (!visited.insert(neighbor).second)
          continue;

        double logP_n =
            logP + log_int[c[j]] - log_int[c[i] + 1] + log_p[i] - log_p[j];
        heap.push({logP_n, neighbor});
      }
    }
  }

  // Build result
  BFSResult result;
  result.prob_left = probs[0] + probs[2];
  result.prob_right = probs[1] + probs[2];
  result.wsum_left.resize(d);
  result.wsum_right.resize(d);
  result.wsum2_left.resize(dk);
  result.wsum2_right.resize(dk);
  for (int i = 0; i < d; ++i) {
    result.wsum_left[i] = wsums[0][i] + wsums[2][i];
    result.wsum_right[i] = wsums[1][i] + wsums[2][i];
  }
  for (int i = 0; i < dk; ++i) {
    result.wsum2_left[i] = wsum2_ptrs[0][i] + wsum2_ptrs[2][i];
    result.wsum2_right[i] = wsum2_ptrs[1][i] + wsum2_ptrs[2][i];
  }
  result.explored_mass = std::min(P_explored, 1.0);
  result.states_explored = states_explored;
  return result;
}

// ─── Mass BFS result ──────────────────────────────────────────────

struct MassBFSResult {
  double explored_mass;
  int64_t states_explored;
};

// ─── GreCov Mass BFS ──────────────────────────────────────────────

static MassBFSResult grecov_mass_bfs_impl(const std::vector<double> &p_raw,
                                          const std::vector<int32_t> &x_obs,
                                          double eps, double tie_margin) {
  int d = static_cast<int>(p_raw.size());
  int n = 0;
  for (auto xi : x_obs)
    n += xi;

  // Stabilise probabilities
  constexpr double MIN_P = 1e-300;
  double p[MAX_D], log_p[MAX_D];
  double p_sum = 0.0;
  for (int i = 0; i < d; ++i) {
    p[i] = std::max(p_raw[i], MIN_P);
    p_sum += p[i];
  }
  for (int i = 0; i < d; ++i) {
    p[i] /= p_sum;
    log_p[i] = std::log(p[i]);
  }

  auto log_fact = log_factorials(n);
  auto log_int = log_integers(n);

  auto log_prob = [&](uint64_t s) -> double {
    double val = log_fact[n];
    for (int i = 0; i < d; ++i) {
      int32_t ci = packed_get(s, i);
      val -= log_fact[ci];
      val += ci * log_p[i];
    }
    return val;
  };

  // Pack x_obs and compute threshold
  int32_t x_obs_arr[MAX_D];
  for (int i = 0; i < d; ++i)
    x_obs_arr[i] = x_obs[i];
  uint64_t x_obs_packed = pack_state(x_obs_arr, d);
  double log_p_obs = log_prob(x_obs_packed);
  double threshold = log_p_obs + tie_margin;

  // Start from approximate mode
  int32_t start_c[MAX_D];
  start_counts(p, n, d, start_c);
  uint64_t start_s = pack_state(start_c, d);
  double start_lp = log_prob(start_s);

  std::priority_queue<Entry> heap;
  std::unordered_set<uint64_t, U64Hash> visited;
  visited.reserve(1 << 16);

  heap.push({start_lp, start_s});
  visited.insert(start_s);

  double mass = 0.0;
  int64_t states_explored = 0;

  while (!heap.empty()) {
    auto top = heap.top();
    heap.pop();
    double logP = top.logP;
    uint64_t state = top.state;

    ++states_explored;

    if (logP <= threshold)
      break;

    mass += std::exp(logP);
    if (1.0 - eps <= mass)
      break;

    // Unpack for neighbor generation
    int32_t c[MAX_D];
    unpack_state(state, c, d);

    for (int j = 0; j < d; ++j) {
      if (c[j] == 0)
        continue;
      for (int i = 0; i < d; ++i) {
        if (i == j)
          continue;

        uint64_t neighbor = packed_transfer(state, i, j);
        if (!visited.insert(neighbor).second)
          continue;

        double logP_n =
            logP + log_int[c[j]] - log_int[c[i] + 1] + log_p[i] - log_p[j];
        heap.push({logP_n, neighbor});
      }
    }
  }

  return {mass, states_explored};
}

// ─── nanobind module ───────────────────────────────────────────────

NB_MODULE(_ext, m) {
  m.doc() = "C++ for the GreCov algorithm";

  m.def(
      "grecov_bfs",
      [](const std::vector<double> &p, const std::vector<double> &v,
         double S_obs, int n, double eps) -> nb::dict {
        auto res = grecov_bfs_impl(p, v, S_obs, n, eps);

        nb::dict d;
        d["prob_left"] = res.prob_left;
        d["prob_right"] = res.prob_right;
        d["wsum_left"] = res.wsum_left;
        d["wsum_right"] = res.wsum_right;
        d["wsum2_left"] = res.wsum2_left;
        d["wsum2_right"] = res.wsum2_right;
        d["explored_mass"] = res.explored_mass;
        d["states_explored"] = res.states_explored;
        return d;
      },
      nb::arg("p"), nb::arg("v"), nb::arg("S_obs"), nb::arg("n"),
      nb::arg("eps") = 1e-4, "Run the GreCov equal-tail BFS algorithm.");

  m.def(
      "grecov_mass_bfs",
      [](const std::vector<double> &p, const std::vector<int> &x_obs,
         double eps, double tie_margin) -> nb::dict {
        std::vector<int32_t> x(x_obs.begin(), x_obs.end());
        auto res = grecov_mass_bfs_impl(p, x, eps, tie_margin);

        nb::dict d;
        d["explored_mass"] = res.explored_mass;
        d["states_explored"] = res.states_explored;
        return d;
      },
      nb::arg("p"), nb::arg("x_obs"), nb::arg("eps") = 1e-3,
      nb::arg("tie_margin") = 1e-8,
      "Run the GreCov mass BFS algorithm.\n\n"
      "Computes pi_> = P({y : P(y|p) > P(x_obs|p)}) with tie margin.");
}
