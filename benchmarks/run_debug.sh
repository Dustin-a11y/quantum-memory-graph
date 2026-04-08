#!/bin/bash
cd /home/dt/Projects/quantum-memory-graph
python3 benchmarks/run_full_benchmark_v2.py > /tmp/benchmark_output.log 2>&1
echo "EXIT CODE: $?" >> /tmp/benchmark_output.log
