# CPU read-bandwidth probe

Synthetic8GiB delivered-load test with AVX512 volatile loads, a warmup and seven timed passes. Compile with Clang18, LLVM OpenMP and mold. Set explicit OMP_PLACES/OMP_NUM_THREADS; reserve unrelated workload CPUs. A NUMA-aware runner must check sufficient memory on each node before execution. This source is specific to the measured two-node server: workerCPU indices modulo256 select node0/1. It binds only its own buffer pages, never migrates existing model pages, and refuses failed bindings. Do not use it unchanged on another topology.

The source emits real64-byte loads; retained disassembly confirmed them. Report best and median, exact affinity, placement and buffer size. Delivered load bytes are not DRAM-controller counters or model throughput. See docs/XOPMC_HARDWARE.md and its sanitized numerical receipt. The published maximum is the best observed among tested thread/placement configurations, not a mathematical hardware bound.
