# Test hardware and bandwidth

Snapshot:28October2026. These are the XopMC development server and measured transfer rates behind the owned native results. They are not minimum requirements or speeds promised for another PC. NVIDIA system indices are resolved at each launch; the table is a dated observation, not a UUID/PCI allowlist.

## CPU, memory and software

| Component | Observed configuration |
| --- | --- |
| CPUs |2 × AMD EPYC9754,128 physical cores per socket |
| Total CPU capacity |256 physical cores,512 logical threads with SMT2 |
| NUMA |2 nodes; node0 CPUs0–127,256–383; node1 CPUs128–255,384–511 |
| ISA |AVX2, AVX-512, VNNI and BF16 available |
| Memory visible to Linux |810,996,461,568 bytes:810.996GB /755.299GiB |
| Linux kernel |7.0.0-38-generic |
| NVIDIA driver |615.71.09, Ada unlock/native-P2P-capable setup |
| Native build |CUDA13.4, Clang18.1.3, mold2.30.0; SM89 |

SMBIOS confirms24×32GB DDR5 modules,12 per socket, with configured and rated speeds4800MT/s. Sustained STREAM results are measured separately below; CPU theoretical memory-channel limits, model file load speed and expert arithmetic counters are not substituted for measured RAM bandwidth.

The measured two-card Swift profile uses32 CPU workers, main CPU127 and interleaved node0/1 memory. That is a deployment choice, not an assertion that all256 physical cores perform every request. More threads or two populated sockets do not automatically make token generation faster.

## Measured system RAM bandwidth

The EPYC9754 specification gives **460.8GB/s per socket**, hence **921.6GB/s aggregate theoretical bandwidth** for two sockets with all12 channels/socket operating atDDR5-4800. This is the interface ceiling, not a guaranteed application result. [AMD EPYC9004 datasheet](https://www.amd.com/content/dam/amd/en/documents/products/epyc/epyc-9004-series-processors-data-sheet.pdf). The kernel EDAC view reports12 populated logical channels and384GiB per socket. A fresh read-only SMBIOS check confirms24×32GB DIMMs,12 per socket, all configured at4800MT/s. This is firmware-reported configured speed, not a live DRAM frequency counter. [Configuration receipt](data/owned-memory-configuration-20261028.json).

### Maximum observed read bandwidth

A separate AVX512 read-only probe reads an8GiB first-touched buffer. Its volatile assembly emits eight64-byte loads per loop; disassembly was retained to check that loads were not optimized away. One warmup precedes seven measured passes. The initial physical-core sweep is shown below. Cores are spread across all32 L3 domains; CPU64–67 remain reserved for HF uploads. This measures delivered CPU load bytes, not DRAM-controller counters, cache-free traffic or model tokens/s.

| Physical threads | Best read GB/s | Median of7 GB/s |
| --- | ---: | ---: |
|64|630.502|607.108|
|128|649.917|643.486|
|192|638.661|615.015|
|252|686.211|666.768|

A subsequent probe explicitly bound each worker's own page-aligned buffer range to its local node, without migrating model pages. It checked per-node free/reclaimable capacity first. The largest observed value is **691.067GB/s**, about75.0% of the theoretical aggregate ceiling. Its252-thread median was657.472GB/s. Extra384/504-thread SMT configurations did not exceed it. These are measured maxima within the tested modes, not proof that no other kernel or firmware configuration can go faster. Independent252-thread local-first-touch runs also showed peaks525.508 and662.282GB/s; placement-sensitive variability remains visible in the receipt. Buffer residency and exact worker placement were checked before timing; no model pages or global memory policy were changed. [Sanitized read receipt](data/owned-ram-read-max-20261028.json).

### STREAM maximum search

The unmodified [STREAM5.10 source](../owned/benchmarks/stream/REFERENCE.md) uses three2GiB double arrays and ten iterations, compiled with Clang18, mold and private LLVM OpenMP. All completed runs passed its numerical validation. Twenty-four CPU/memory-placement configurations were tested:16–128 physical threads per socket and32–252 across both. The best dual-socket mode uses252 physical threads, spread across L3 domains with local first-touch allocation. HF CPUs64–67 are excluded.

| Best selected mode | Copy GB/s | Scale GB/s | Add GB/s | Triad GB/s |
| --- | ---: | ---: | ---: | ---: |
|Socket0,124threads|330.055|241.713|265.636|268.324|
|Socket1,128threads|329.066|239.649|260.048|265.068|
|Both sockets,252threads|510.092|471.717|510.458|524.793|

STREAM reports logical read/write decimal GB/s; write allocation and other memory-controller traffic are not included in that logical byte count. Its **524.793GB/s Triad** therefore measures a different workload from the read-only probe. Three additional independent best-mode runs produced408.626,520.427 and522.965GB/s Triad; the slower run remains in the receipt rather than being discarded. The existing model RAM keeper identity was unchanged across the tests. [Sanitized STREAM receipts](data/owned-ram-bandwidth-maximum-20261028.json).

An initial test with32 contiguous cores per socket measured only187GB/s combined Triad. It was not the server's bandwidth ceiling: distributing work across L3 domains, increasing thread count and using local first touch raised the observed peak substantially. CPU expert scheduling should be evaluated separately; the synthetic winner's252 threads must not automatically replace the model's qualified32-worker policy.

### Why the measured rate is below921.6GB/s

The theoretical figure multiplies channels, transfer rate and data width. It assumes continuous useful data transfer; sustained software bandwidth also depends on memory commands, refresh, bank/row behavior, core-to-I/O-die paths, NUMA placement and the instruction stream. A difference from the interface ceiling alone does not identify a failed channel or a slow DIMM.

Two effects were directly tested here. The first STREAM binary had no non-temporal stores. A separately labeled tuned variant with AVX2 streaming stores and per-worker store fences raised maximum Triad from524.793 to683.468GB/s, retaining STREAM validation. Temporal-store write allocation is a supported explanation: its extra traffic is absent from STREAM's logical byte count. Separately, explicit local placement plus hugepage advice for the read probe's own buffer raised the observed read peak to699.330GB/s. These changes affect benchmark buffers only.

AMD's [HPC tuning guide58002, table8-7](https://docs.amd.com/api/khub/documents/NgOfoW49HKdzztTeLBbekA/content) reports up to787.422GB/s STREAM on a different dual9754 system using NPS4, optimized stores and its stated setup. This machine exposes one Linux NUMA node per socket. Its remaining difference from that reference is not fully attributed: a controlled NPS/firmware comparison and memory-controller counters were not collected. No BIOS, governor, power, driver or global hugepage setting was changed. [Diagnostic receipt](data/owned-ram-diagnosis-20261028.json).

## Installed GPU topology

All five currently observed cards are GeForce RTX4090 (AdaSM89), with24,564MiB reported by NVML per card. Power limits are500W; this is a limit, not measured continuous consumption. No power setting was changed for this documentation.

| Current system index | Negotiated PCIe link | GPU-local NUMA node | Current use in the owned optimization |
| --- | --- | ---: | --- |
|GPU0|PCIe4.0 ×16|0|Primary selected pair|
|GPU1|PCIe4.0 ×16|0|Primary selected pair|
|GPU2|PCIe4.0 ×16|1|Preserved three/four-card research|
|GPU3|PCIe4.0 ×16|1|Preserved four-card research|
|GPU4|PCIe4.0 ×4|1|Not included in these model measurements|

GPU0↔GPU1 and GPU2↔GPU3 are `NODE` paths in `nvidia-smi topo -m`. Cross-socket pairs report `SYS`: they traverse the SMP/NUMA interconnect as well as PCIe. There is no NVLink link in this topology. This matters for small, frequent activation exchanges and shared ingress bandwidth; summing GPU peak compute or VRAM capacities does not predict one-request speed.

The policy API accepts up to7 selected compatible GPUs. Five are currently observed; seven-card model execution and bandwidth have not been measured. Multi-card compute selection is separate from system-wide driver P2P capability.

## Link specifications versus measurements

PCIe4.0 uses16GT/s per lane and128b/130b encoding. Before transaction overhead, ×16 is31.51GB/s in each direction and ×4 is7.88GB/s in each direction. Full duplex does not mean a single one-way transfer gets the sum of both directions.

The RTX4090 specification lists384-bit GDDR6X at21Gb/s, for1008GB/s theoretical local VRAM bandwidth. This is GPU-local memory bandwidth, not host→GPU or peer→peer bandwidth. See [NVIDIA Ada specifications, appendixA](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

### Fresh selected-pair measurement

The following was measured on28October2026 with current GPU0/1 idle, native peer access admitted in both directions, and no loaded GPU model. Component source: [peer_bandwidth.cpp](../owned/benchmarks/peer_bandwidth.cpp); [sanitized result](data/owned-bandwidth-20261028.json).

| Transfer | Block size | Repetitions | Measured decimal GB/s |
| --- | ---: | ---: | ---: |
|GPU0→GPU1 native P2P|32MiB|32|26.257|
|GPU1→GPU0 native P2P|32MiB|32|26.245|
|GPU0→GPU1 native P2P|256KiB|256|17.716|
|GPU1→GPU0 native P2P|256KiB|256|17.716|
|Pinned host→GPU0|32MiB|32|26.826|
|GPU0→pinned host|32MiB|32|26.358|
|Pinned host→GPU1|32MiB|32|26.836|
|GPU1→pinned host|32MiB|32|26.368|

Method: four warmups, then enqueue the stated copy count and synchronize the owning stream; elapsed time is host wall time. Directions are tested sequentially.256KiB peer copies took about14.8µs each including this loop's enqueue/synchronization overhead; this is not an isolated hardware latency number. The test checks payload edges. It does not run model kernels or establish generation tokens/s. Host buffers are pinned, and the controller CPU was127 on NUMA0; both selected GPUs are on node0.

## Historical scaling evidence

A separate26October four-card transfer experiment measured about26.15–26.25GB/s for individual32MiB peer directions. Three simultaneous senders into one receiver reached about26.4GB/s **in aggregate**, rather than three times26GB/s. Its receipt did not contain a complete current driver/boot/link snapshot; it is historical evidence of a shared-ingress bottleneck, not a new five-card measurement. Do not combine that aggregate with the fresh one-way table as though they were one experiment.

Older pre-P2P measurements observed about6.7GB/s on a PCIe4.0 ×4 card. That is a historical host-transfer result, not a fresh benchmark of currentGPU4. CurrentGPU2–4 were not used by the new two-card copy test.

## Native P2P and reproducibility

This server uses an Ada unlock/P2P-capable driver setup; the user's original unlock project is [gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada). Driver capability is system-wide, while peer mappings belong to each selected CUDA process. The code checks all required selected directions before model allocation/capture and refuses startup when native P2P is unavailable. It does not install a driver, reset a GPU or silently stage inter-GPU tensors through host memory.

Keep the hardware snapshot, selected indices, current topology, driver, context fill, KV/MTP settings, worker placement and transfer method with a benchmark. See [full model results](XOPMC_RESULTS.md), [native P2P policy](XOPMC_P2P.md) and [model downloads](XOPMC_MODELS.md).
