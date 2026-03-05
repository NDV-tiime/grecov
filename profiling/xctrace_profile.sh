#!/usr/bin/env bash
# CPU-profile the C++ BFS code using macOS xctrace.
#
# Usage:
#   ./profiling/xctrace_profile.sh                  # profile default script
#   ./profiling/xctrace_profile.sh my_script.py     # profile custom script
#   ./profiling/xctrace_profile.sh --build-only      # just rebuild with symbols
#
# The script will:
#   1. Rebuild the extension with debug symbols
#   2. Record a Time Profiler trace
#   3. Export and print the top hotspot functions
#   4. Rebuild for production (no profiling overhead)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRIPT="${1:-profiling/profile_bfs.py}"
TRACE="profile.trace"
SAMPLES="samples.xml"
PYTHON=$(python -c "import sys; print(sys.executable)")
TIME_LIMIT="${XCTRACE_TIME_LIMIT:-30s}"

# ── Step 1: Build with profiling symbols ─────────────────────────
echo "==> Building with profiling symbols..."
pip install -e . -q \
  --config-settings=cmake.define.GRECOV_PROFILE=ON \
  --config-settings=cmake.define.CMAKE_BUILD_TYPE=RelWithDebInfo

if [[ "${1:-}" == "--build-only" ]]; then
  echo "Done (build only)."
  exit 0
fi

# ── Step 2: Record trace ─────────────────────────────────────────
echo "==> Recording trace (time limit: $TIME_LIMIT)..."
rm -rf "$TRACE"
xcrun xctrace record \
  --template "Time Profiler" \
  --time-limit "$TIME_LIMIT" \
  --output "$TRACE" \
  --launch -- "$PYTHON" "$SCRIPT"

# ── Step 3: Export and analyze ───────────────────────────────────
echo "==> Exporting samples..."
xcrun xctrace export \
  --input "$TRACE" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  --output "$SAMPLES"

echo ""
echo "=== Top 30 hotspot functions ==="
grep -o 'name="[^"]*"' "$SAMPLES" \
  | sed 's/name="//; s/"//' \
  | sort | uniq -c | sort -rn | head -30

# ── Step 4: Rebuild for production ───────────────────────────────
echo ""
echo "==> Rebuilding for production..."
pip install -e . -q

echo ""
echo "Done. Trace saved to $TRACE, samples to $SAMPLES."
