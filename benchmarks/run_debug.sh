#!/bin/bash
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
python3 benchmarks/run_full_benchmark_v2.py > /tmp/benchmark_output.log 2>&1
echo "EXIT CODE: $?" >> /tmp/benchmark_output.log
