# Profiling

## Benchmarks

```bash
python profiling/bench_bfs.py --bfs      # BFS microbenchmarks
python profiling/bench_bfs.py --solver   # full solver benchmark
python profiling/bench_bfs.py            # both
```

## CPU profiling (macOS, xctrace)

```bash
# Profile the default BFS workload (builds with symbols, records, analyzes, rebuilds for prod)
./profiling/xctrace_profile.sh

# Profile a custom script
./profiling/xctrace_profile.sh my_script.py

# Just rebuild with profiling symbols (e.g. to attach Instruments.app manually)
./profiling/xctrace_profile.sh --build-only

# Override time limit (default 30s)
XCTRACE_TIME_LIMIT=60s ./profiling/xctrace_profile.sh
```

Output files (`profile.trace`, `samples.xml`) are gitignored.
