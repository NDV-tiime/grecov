// C++ implementation of the GreCov BFS algorithm
//
// Optimised best-first enumeration of multinomial count vectors in decreasing
// probability order.  Key optimisations:
// - Pack d-dimensional count vectors into a single uint64_t (big-endian,
//   8 bits per dimension, d<=8, counts<=255) for cheap hashing & comparison.
// - Open-addressing hash set (FlatHashSet) for cache-friendly lookups.
// - Template on D (dimension) so inner loops are fully unrolled at compile
// time.
// - Precomputed log(i) lookup table to avoid repeated std::log() calls.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

namespace nb = nanobind;

// ─── Packed state representation ──────────────────────────────────
// Each count occupies 8 bits, stored big-endian so that uint64_t numeric
// comparison matches lexicographic vector comparison (needed for heap
// tiebreaker determinism).

static constexpr int SHIFT[8] = {56, 48, 40, 32, 24, 16, 8, 0};
static constexpr uint64_t DELTA[8] = {
    1ULL << 56, 1ULL << 48, 1ULL << 40, 1ULL << 32,
    1ULL << 24, 1ULL << 16, 1ULL << 8,  1ULL << 0,
};

template <int D> inline uint64_t pack_state(const int *c) {
  uint64_t s = 0;
  for (int i = 0; i < D; ++i)
    s |= static_cast<uint64_t>(c[i]) << SHIFT[i];
  return s;
}

template <int D> inline void unpack_state(uint64_t s, int *c) {
  for (int i = 0; i < D; ++i)
    c[i] = static_cast<int>((s >> SHIFT[i]) & 0xFF);
}

// ─── Open-addressing hash set ─────────────────────────────────────

class FlatHashSet {
  static constexpr uint64_t EMPTY = ~uint64_t(0);

  std::vector<uint64_t> slots_;
  size_t mask_;
  size_t size_;

  static size_t hash(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<size_t>(x);
  }

public:
  explicit FlatHashSet(size_t initial_cap = 1 << 16)
      : slots_(initial_cap, EMPTY), mask_(initial_cap - 1), size_(0) {}

  bool insert(uint64_t key) {
    if (__builtin_expect(size_ * 4 >= slots_.size() * 3, 0))
      grow();
    size_t idx = hash(key) & mask_;
    while (true) {
      uint64_t slot = slots_[idx];
      if (__builtin_expect(slot == EMPTY, 1)) {
        slots_[idx] = key;
        ++size_;
        return true;
      }
      if (slot == key)
        return false;
      idx = (idx + 1) & mask_;
    }
  }

private:
  void grow() {
    size_t new_cap = slots_.size() * 2;
    std::vector<uint64_t> old_slots(new_cap, EMPTY);
    std::swap(old_slots, slots_);
    mask_ = new_cap - 1;
    size_ = 0;
    for (auto v : old_slots) {
      if (v != EMPTY)
        insert(v);
    }
  }
};

// ─── Heap entry ───────────────────────────────────────────────────
// Max-heap by logP; on tie, smaller packed state = higher priority
// (matches Python's min-heap on (-logP, counts_tuple)).

using Entry = std::pair<double, uint64_t>;

struct EntryCompare {
  bool operator()(const Entry &a, const Entry &b) const {
    return a.first < b.first || (a.first == b.first && a.second > b.second);
  }
};

// ─── Precomputed tables ───────────────────────────────────────────

static std::vector<double> log_factorials(int n) {
  std::vector<double> lf(n + 1, 0.0);
  for (int i = 1; i <= n; ++i)
    lf[i] = lf[i - 1] + std::log(static_cast<double>(i));
  return lf;
}

// log(i) for i in [0..n], with log(0) = 0 (never used)
static std::vector<double> log_integers(int n) {
  std::vector<double> li(n + 1, 0.0);
  for (int i = 1; i <= n; ++i)
    li[i] = std::log(static_cast<double>(i));
  return li;
}

// ─── Balanced rounding (largest-remainder method) ──────────────────

template <int D> static uint64_t start_counts(const double *p, int n) {
  int counts[D];
  double frac[D];
  int total = 0;
  for (int i = 0; i < D; ++i) {
    double x = p[i] * n;
    counts[i] = static_cast<int>(std::floor(x));
    frac[i] = x - counts[i];
    total += counts[i];
  }
  int remainder = n - total;
  int idx[D];
  for (int i = 0; i < D; ++i)
    idx[i] = i;
  std::sort(idx, idx + D, [&frac](int a, int b) { return frac[a] > frac[b]; });
  for (int r = 0; r < remainder; ++r)
    counts[idx[r]] += 1;
  return pack_state<D>(counts);
}

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

// ─── Templated tail BFS ────────────────────────────────────────────

template <int D>
__attribute__((noinline)) static BFSResult
grecov_bfs_d(const double *p, const double *v, double S_obs, int n,
             double eps) {
  double log_p[D];
  for (int i = 0; i < D; ++i)
    log_p[i] = std::log(p[i]);

  auto log_fact = log_factorials(n);
  auto li = log_integers(n);

  auto log_prob = [&](uint64_t state) -> double {
    int c[D];
    unpack_state<D>(state, c);
    double val = log_fact[n];
    for (int i = 0; i < D; ++i) {
      val -= log_fact[c[i]];
      val += c[i] * log_p[i];
    }
    return val;
  };

  uint64_t start_state = start_counts<D>(p, n);
  double start_lp = log_prob(start_state);

  std::vector<Entry> heap_storage;
  heap_storage.reserve(1 << 16);
  std::priority_queue<Entry, std::vector<Entry>, EntryCompare> heap(
      EntryCompare{}, std::move(heap_storage));
  FlatHashSet visited;

  heap.push({start_lp, start_state});
  visited.insert(start_state);

  double P_explored = 0.0;
  constexpr int dk = D * D;
  double probs[3] = {};
  double wsums[3][D] = {};
  double wsum2s[3][dk] = {};
  int64_t states_explored = 0;

  auto accumulate = [&](int side, double P_state, const int *c) {
    probs[side] += P_state;
    auto &ws = wsums[side];
    auto &ws2 = wsum2s[side];
    for (int i = 0; i < D; ++i) {
      double pci = P_state * c[i];
      ws[i] += pci;
      ws2[i * D + i] += pci * c[i];
      for (int j = i + 1; j < D; ++j) {
        double val = pci * c[j];
        ws2[i * D + j] += val;
        ws2[j * D + i] += val;
      }
    }
  };

  while (!heap.empty()) {
    auto [logP, state] = heap.top();
    heap.pop();

    ++states_explored;
    double P_state = std::exp(logP);
    P_explored += P_state;

    int c[D];
    unpack_state<D>(state, c);

    double s_val = 0.0;
    for (int i = 0; i < D; ++i)
      s_val += c[i] * v[i];

    constexpr int L = 0, R = 1, E = 2;
    if (s_val < S_obs)
      accumulate(L, P_state, c);
    else if (s_val > S_obs)
      accumulate(R, P_state, c);
    else
      accumulate(E, P_state, c);

    if (1.0 - eps <= P_explored)
      break;

    for (int j = 0; j < D; ++j) {
      if (c[j] == 0)
        continue;
      for (int i = 0; i < D; ++i) {
        if (i == j)
          continue;
        uint64_t neighbor = state + DELTA[i] - DELTA[j];
        if (!visited.insert(neighbor))
          continue;
        double logP_n = logP + li[c[j]] - li[c[i] + 1] + log_p[i] - log_p[j];
        heap.push({logP_n, neighbor});
      }
    }
  }

  constexpr int L = 0, R = 1, E = 2;
  BFSResult result;
  result.prob_left = probs[L] + probs[E];
  result.prob_right = probs[R] + probs[E];
  result.wsum_left.resize(D);
  result.wsum_right.resize(D);
  result.wsum2_left.resize(dk);
  result.wsum2_right.resize(dk);
  for (int i = 0; i < D; ++i) {
    result.wsum_left[i] = wsums[L][i] + wsums[E][i];
    result.wsum_right[i] = wsums[R][i] + wsums[E][i];
  }
  for (int i = 0; i < dk; ++i) {
    result.wsum2_left[i] = wsum2s[L][i] + wsum2s[E][i];
    result.wsum2_right[i] = wsum2s[R][i] + wsum2s[E][i];
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

// ─── Templated mass BFS ───────────────────────────────────────────

template <int D>
__attribute__((noinline)) static MassBFSResult
grecov_mass_bfs_d(const double *p, const int *x_obs, int n, double eps,
                  double tie_margin) {
  double log_p[D];
  for (int i = 0; i < D; ++i)
    log_p[i] = std::log(p[i]);

  auto log_fact = log_factorials(n);
  auto li = log_integers(n);

  auto log_prob = [&](const int *c) -> double {
    double val = log_fact[n];
    for (int i = 0; i < D; ++i) {
      val -= log_fact[c[i]];
      val += c[i] * log_p[i];
    }
    return val;
  };

  double log_p_obs = log_prob(x_obs);
  double threshold = log_p_obs + tie_margin;

  uint64_t start_state = start_counts<D>(p, n);

  // Compute start log-prob via unpack
  int sc[D];
  unpack_state<D>(start_state, sc);
  double start_lp = log_prob(sc);

  std::vector<Entry> heap_storage;
  heap_storage.reserve(1 << 16);
  std::priority_queue<Entry, std::vector<Entry>, EntryCompare> heap(
      EntryCompare{}, std::move(heap_storage));
  FlatHashSet visited;

  heap.push({start_lp, start_state});
  visited.insert(start_state);

  double mass = 0.0;
  int64_t states_explored = 0;

  while (!heap.empty()) {
    auto [logP, state] = heap.top();
    heap.pop();

    ++states_explored;

    if (logP <= threshold)
      break;

    mass += std::exp(logP);

    if (1.0 - eps <= mass)
      break;

    int c[D];
    unpack_state<D>(state, c);

    for (int j = 0; j < D; ++j) {
      if (c[j] == 0)
        continue;
      for (int i = 0; i < D; ++i) {
        if (i == j)
          continue;
        uint64_t neighbor = state + DELTA[i] - DELTA[j];
        if (!visited.insert(neighbor))
          continue;
        double logP_n = logP + li[c[j]] - li[c[i] + 1] + log_p[i] - log_p[j];
        heap.push({logP_n, neighbor});
      }
    }
  }

  return {mass, states_explored};
}

// ─── Dispatch by dimension ────────────────────────────────────────

static BFSResult grecov_bfs_dispatch(const std::vector<double> &p_raw,
                                     const std::vector<double> &v, double S_obs,
                                     int n, double eps) {
  int d = static_cast<int>(p_raw.size());

  // Stabilise probabilities
  constexpr double MIN_P = 1e-300;
  std::vector<double> p(d);
  double p_sum = 0.0;
  for (int i = 0; i < d; ++i) {
    p[i] = std::max(p_raw[i], MIN_P);
    p_sum += p[i];
  }
  for (int i = 0; i < d; ++i)
    p[i] /= p_sum;

  const double *pp = p.data();
  const double *vp = v.data();

#define DISPATCH_BFS(D_VAL)                                                    \
  case D_VAL:                                                                  \
    return grecov_bfs_d<D_VAL>(pp, vp, S_obs, n, eps);

  switch (d) {
    DISPATCH_BFS(2)
    DISPATCH_BFS(3)
    DISPATCH_BFS(4)
    DISPATCH_BFS(5)
    DISPATCH_BFS(6)
    DISPATCH_BFS(7)
    DISPATCH_BFS(8)
  default:
    throw std::runtime_error("dimension must be between 2 and 8");
  }
#undef DISPATCH_BFS
}

static MassBFSResult grecov_mass_bfs_dispatch(const std::vector<double> &p_raw,
                                              const std::vector<int32_t> &x_obs,
                                              double eps, double tie_margin) {
  int d = static_cast<int>(p_raw.size());
  int n = 0;
  for (auto xi : x_obs)
    n += xi;

  // Stabilise probabilities
  constexpr double MIN_P = 1e-300;
  std::vector<double> p(d);
  double p_sum = 0.0;
  for (int i = 0; i < d; ++i) {
    p[i] = std::max(p_raw[i], MIN_P);
    p_sum += p[i];
  }
  for (int i = 0; i < d; ++i)
    p[i] /= p_sum;

  std::vector<int> x(x_obs.begin(), x_obs.end());
  const double *pp = p.data();
  const int *xp = x.data();

#define DISPATCH_MASS_BFS(D_VAL)                                               \
  case D_VAL:                                                                  \
    return grecov_mass_bfs_d<D_VAL>(pp, xp, n, eps, tie_margin);

  switch (d) {
    DISPATCH_MASS_BFS(2)
    DISPATCH_MASS_BFS(3)
    DISPATCH_MASS_BFS(4)
    DISPATCH_MASS_BFS(5)
    DISPATCH_MASS_BFS(6)
    DISPATCH_MASS_BFS(7)
    DISPATCH_MASS_BFS(8)
  default:
    throw std::runtime_error("dimension must be between 2 and 8");
  }
#undef DISPATCH_MASS_BFS
}

// ─── nanobind module ───────────────────────────────────────────────

NB_MODULE(_ext, m) {
  m.doc() = "C++ for the GreCov algorithm";

  m.def(
      "grecov_bfs",
      [](const std::vector<double> &p, const std::vector<double> &v,
         double S_obs, int n, double eps) -> nb::dict {
        auto res = grecov_bfs_dispatch(p, v, S_obs, n, eps);

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
        auto res = grecov_mass_bfs_dispatch(p, x, eps, tie_margin);

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
