<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h2 align="center">Strata-Server</h2>
<h3 align="center">Large models. Your hardware. Native CPU and GPU execution.</h3>
<p align="center">Swift Flash Next and DeepSeek, with shared expert memory and direct multi-GPU transport.</p>

<p align="center">
  <a href="#what-is-strata-server">Overview</a> ·
  <a href="#models">Models</a> ·
  <a href="#benchmarks"><b>Benchmarks</b></a> ·
  <a href="#test-hardware">Hardware</a> ·
  <a href="#build-and-run">Build</a> ·
  <a href="#guides">Guides</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/XopMC/Strata-Server" alt="MIT license"></a>
  <a href="https://huggingface.co/XopMC513"><img src="https://img.shields.io/badge/Hugging_Face-Models-FFD21E" alt="Our models on Hugging Face"></a>
  <a href="https://github.com/Niko1221/Strata"><img src="https://img.shields.io/badge/Based_on-Strata-0ab887" alt="Based on Strata"></a>
</p>

**Free, open source, and yours to run.** Strata-Server brings our native Swift and DeepSeek backends into one source checkout, with explicit choices for the model, GPU count and context. It is built on [Strata](https://github.com/Niko1221/Strata); the original app, Terminal and documentation remain available.

## What is Strata-Server?

The focus is large Mixture-of-Experts models on a CPU with plenty of RAM and one or more NVIDIA GPUs. Keep experts in system memory, retain useful weights on the GPUs, and give each part of the machine a defined role.

| What is included | What it does |
| --- | --- |
| **Native Swift Flash Next and DeepSeek providers** | Keep the matching loaders, kernels, attention state and speculative decoding for each model. |
| **One build entry** | Select the model family, GPU count and context with CMake flags. Numerical implementations keep their own matching headers and ABI. |
| **Shared expert RAM** | Compatible native workers can attach to one sealed, read-only expert bank instead of loading the expert payload again. Request KV and recurrent state remain separate. |
| **Required native P2P** | Communicating GPUs use direct GPU-resident transport. Startup refuses unavailable peer access; it does not silently switch to host staging. |
| **NVFP4 execution on Ada** | Custom CPU and GPU paths run the selected mixed NVFP4 package on RTX 4090. Ada does not acquire Blackwell FP4 Tensor Cores from a weight format. |
| **Verified MTP** | The target model checks draft tokens and owns acceptance and rollback. Draft speed is not presented as final generation speed. |
| **Fresh device and memory discovery** | GPU indices resolve at launch. NUMA interleave follows current allowed memory nodes, including changed NPS layouts, rather than a saved UUID or two-node list. |
| **Measured source reconstruction** | The two-GPU Swift provider has been rebuilt from source and checked against the original backend on a filled 512K context. |

All-expert, full-attention comparison profiles are retained. The selected source profiles do not introduce expert skipping, a shorter attention window or a lower-bit replacement for the published weights. Useful experiments and architecture work remain separate from qualified defaults.

## Models

Our checkpoints are published under **[XopMC513 on Hugging Face](https://huggingface.co/XopMC513)**. Weights are separate downloads; this Git repository contains source, profiles and documentation.

| Model | Published versions | Native work in this fork |
| --- | --- | --- |
| **[Swift 1.5 Qwen3.8 Flash Next Heretic](https://huggingface.co/XopMC513/Swift-1.5-Qwen3.8-Flash-Next-Heretic)** | Own gentle-Heretic original-precision weights, **Q4_K_M**, and a **custom mixed NVFP4** package | CPU expert execution, shared RAM, GPU expert residency, prefill, MTP and 1–4 GPU source profiles |
| **[DeepSeek V4.1 Flash 1017 Heretic](https://huggingface.co/XopMC513/DeepSeek-V4.1-Flash-1017-Heretic)** | Own gentle-Heretic original-precision checkpoint and **mixed NVFP4-based quantization** | Separate native DeepSeek implementation, compressed attention, CPU/GPU experts and two-GPU source profiles |

DeepSeek's original-precision checkpoint preserves its original tensor formats, including FP8/MXFP4 where present; it is not described as an all-BF16 model. Custom native packages need their matching loader. An `.safetensors` or `.gguf` extension alone does not establish compatible arithmetic.

[Formats, assets and compatibility →](docs/XOPMC_MODELS.md)

## Benchmarks

**These are our local measurements**, on the dual-EPYC / RTX 4090 server below. Prompt speed counts **new input tokens**. Long-context rows use genuinely filled contexts, not just a large context allocation. Speeds are for one request, not the sum of several requests.

### Filled contexts

| Model and backend | GPUs | New input / allocated context | Generation | New prompt |
| --- | ---: | ---: | ---: | ---: |
| Swift mixed NVFP4, **unified source rebuild** | 2 | 520,000 / 524,288 | **192.34 tok/s** | **9,265 tok/s** |
| Swift mixed NVFP4, original qualified implementation | 2 | 1,040,000 / 1,048,576 | **187.64 tok/s** | **7,753 tok/s** |
| Swift mixed NVFP4, original prefill-oriented profile | 3 | 260,000 / 262,144 | **177.89 tok/s** | **11,457 tok/s** |
| Swift mixed NVFP4, original prefill-oriented profile | 3 | 520,000 / 524,288 | **185.38 tok/s** | **12,321 tok/s** |
| DeepSeek 1017 mixed, original native API profile | 2 | 261,568 / 262,144 | **54.69 tok/s** | **4,936 tok/s** |
| DeepSeek 1017 mixed, original native API profile | 2 | 523,712 / 524,288 | **56.64 tok/s** | **5,590 tok/s** |

The Swift rows generated 1,024 real output tokens; DeepSeek rows generated 512. The listed long-context configurations are individual runs. The three-GPU prefill result and the two-GPU generation result are different profiles: their best numbers are not combined into an imaginary configuration. More cards did not automatically improve generation. Four-GPU work is retained as experimental source, not advertised as a faster default.

### Small fill inside a long allocation

| Model and backend | New input / allocated context | Generation | New prompt |
| --- | ---: | ---: | ---: |
| Swift mixed NVFP4, unified two-GPU source rebuild | 16,384 / 524,288 | **225.05 tok/s** | **3,298 tok/s** |

This is a fresh short request in the same process, not a filled 512K result. It is shown separately because prompt length changes the workload.

### What was checked in the unified rebuild?

The final two-GPU source build matched the original control's **initial logits, prefill geometry and all 1,024 output token IDs**, for both 520,000-token and 16,384-token inputs. Expert cache maps stayed the same. The existing shared bank was reused with **zero expert payload bytes reloaded**. Native shutdown drained the captured resource pools.

The original control measured 192.22 generation / 9,244 new-prompt tok/s on the long fill. The rebuilt provider measured 192.34 / 9,265. This establishes reconstruction parity on these tests; the small timing difference is not promoted as a new optimization gain.

Historical single-GPU Swift medians were **173.26 / 181.79 / 175.29 generation tok/s** and **5,297 / 4,752 / 3,833 new-prompt tok/s** at 256K / 512K / 1M. They used a different adaptive profile and are not a controlled one-to-three-card scaling comparison.

[Full results, settings and limits →](docs/XOPMC_RESULTS.md) · [Machine-readable source qualification →](owned/swift/QUALIFICATION.json)

## Test hardware

Snapshot: **28 October 2026**. Hardware indices are dated observations, not permanent allowlists.

| Component | Development server |
| --- | --- |
| CPU | **2 × AMD EPYC 9754** — 256 physical cores, 512 logical threads |
| Memory | **24 × 32 GB DDR5**, 12 populated channels per socket; all DIMMs report **4,800 MT/s configured speed** |
| Memory visible to Linux | **810.996 GB / 755.299 GiB** |
| GPUs | **5 × GeForce RTX 4090**, 24,564 MiB reported per card |
| GPU links | Current GPU0–3: **PCIe 4.0 ×16**; GPU4: **PCIe 4.0 ×4** |
| Locality | GPU0/1 on socket 0; GPU2/3/4 on socket 1; cross-socket traffic traverses the CPU interconnect |
| GPU transport | Native P2P-capable Ada setup; **no NVLink** |
| GPU power limit | 500 W reported; no power change was made for these measurements |
| Software | Linux 7.0.0-38, NVIDIA 615.71.09, CUDA 13.4, **Clang 18 + mold**, SM89 |

### Measured bandwidth

| Path or workload | Result | Meaning |
| --- | ---: | --- |
| **System RAM, read-only peak** | **699.33 GB/s** | Actual load-byte throughput, locally placed test buffers with hugepage advice |
| **System RAM, tuned STREAM Triad** | **683.47 GB/s** | Validated logical read/write throughput with non-temporal stores |
| System RAM, theoretical aggregate ceiling | 921.6 GB/s | 2 sockets × 12 channels × 4,800 MT/s × 8 bytes; not a sustained application result |
| **GPU0 → GPU1 native P2P** | **26.26 GB/s** | 32 MiB blocks, sequential one-way test |
| **GPU1 → GPU0 native P2P** | **26.25 GB/s** | Same method, reverse direction |
| Pinned host → GPU0 / GPU1 | 26.83 / 26.84 GB/s | Measured host transfers, not system RAM bandwidth |
| RTX 4090 local VRAM, specification | 1,008 GB/s | Theoretical GPU-local bandwidth, not PCIe or peer-copy speed |

RAM tests searched thread counts, L3-domain placement and NUMA policy; the largest observed values are not a proof of an absolute hardware maximum. The copy tests and RAM tests are component measurements, not model token rates. Methods, repeats, variability and sanitized receipts are linked below.

[Hardware, NUMA topology and bandwidth methods →](docs/XOPMC_HARDWARE.md)

## Build and run

The owned providers are opt-in. Building the upstream project alone does not select one of these native closures.

```sh
git clone https://github.com/XopMC/Strata-Server.git
cd Strata-Server

cmake -S owned -B build-owned -G Ninja \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_OWNED_FAMILY=swift \
  -DSTRATA_OWNED_GPU_COUNT=2 \
  -DSTRATA_OWNED_CONTEXT=524288

cmake --build build-owned --target strata_owned_swift -j8
```

### Model and GPU selection

| CMake choice | Source profile | Current qualification |
| --- | --- | --- |
| `swift`, 1 GPU, 256K / 512K / 1M | Context-specific single-GPU providers | Source imported; rebuilt-model qualification pending |
| `swift`, 2 GPUs, 512K / 1M | Selected two-GPU provider | Source build and filled-512K parity checked; rebuilt 1M qualification separate |
| `swift`, 3 GPUs, 256K / 512K | Selected prefill-oriented provider | Source imported; rebuilt-model qualification pending |
| `swift`, 4 GPUs | Preserved experimental provider | Requires `STRATA_OWNED_EXPERIMENTAL=ON`; not a production winner |
| `deepseek`, 2 GPUs, 256K / 512K | Context-specific DeepSeek provider | Source checks passed; clean build and common serving-factory qualification pending |

The shared peer-policy API handles up to seven selected compatible devices. This does not mean seven-GPU model execution has been implemented or measured. Unqualified launch profiles are disabled, and unsupported choices fail explicitly.

Provide compatible existing model assets, then inspect the native launch plan:

```sh
python3 owned/launch.py \
  --profile swift-nvfp4-two --gpus 0,1 --context 512k \
  --pack /models/swift/native-pack-experts-v1 \
  --common /models/swift/common.gguf \
  --mtp /models/swift/mtp-production-q4 \
  --expert-profile /models/swift/expert-profile.bin \
  --plan
```

`--gpus` means **current NVIDIA system indices**, resolved afresh. `--plan` loads no CUDA or model weights. `--native-server` selects the native stdin protocol; it is not an HTTP listener. The original Strata app and compatible APIs remain in the repository, while integration of every owned provider with a common HTTP factory is separate work.

A valid `--shared-expert-arena` can reuse an existing sealed bank. It does not share request KV or recover a pre-reboot bank. After a BIOS NPS change, CPU placement and memory-node admission must be checked again; the selector discovers current allowed memory nodes instead of assuming `0,1` is the entire server.

[Complete source build and runtime contract →](owned/README.md) · [Profile registry →](owned/profiles/README.md) · [P2P admission →](docs/XOPMC_P2P.md)

## Research and retained work

The upstream audit covered **367 commits and 534 changed paths** through Strata 0.2.7. Related work includes Project Maya, 1Cat-vLLM and Uzu. Useful directions include shape-specific kernels, compact activation exchange, exact communication overlap, speculative-tree budgets and explicit recurrent-state commit rules.

Candidates are not silently enabled because another project reports a speedup. Numerical order, supported formats, graph lifetimes and actual whole-model results still matter. Uzu is primarily Metal/Apple work; it does not provide a ready Ada CUDA NVFP4/P2P backend.

[Upstream review and adjacent-project findings →](docs/XOPMC_RESEARCH.md)

## Guides

[Model downloads](docs/XOPMC_MODELS.md) · [Native results](docs/XOPMC_RESULTS.md) · [Hardware and RAM bandwidth](docs/XOPMC_HARDWARE.md) · [Build flags](owned/README.md) · [Native P2P](docs/XOPMC_P2P.md) · [Source status](owned/profiles/README.md)

<details>
<summary><b>Original Strata app, Terminal and platform guides</b></summary>

The original Strata features and platform documentation are preserved. Upstream release packages do not include these newly reconstructed owned native providers.

- [Original Strata releases](https://github.com/Niko1221/Strata/releases/latest)
- [Desktop installation](docs/INSTALL.md) · [Terminal guide](docs/TERMINAL_LAUNCH.md)
- [OpenAI- and Anthropic-compatible API connections](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect)
- [Original source installation](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md)
- [AMD](docs/AMD_HIP.md) · [Intel Arc](docs/INTEL_ARC.md) · [Apple Silicon](docs/MAC.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md) · [Contributing](CONTRIBUTING.md)

</details>

## Credits and licenses

**Strata-Server builds on [Strata by Niko1221 and contributors](https://github.com/Niko1221/Strata).** The original visual style, app and Terminal are retained. This fork's native work and local measurements are maintained by [XopMC](https://github.com/XopMC); checkpoints are at [Hugging Face](https://huggingface.co/XopMC513).

[MIT license](LICENSE) · [Original credits](docs/HOW_IT_WORKS.md#credits) · [Swift third-party notices](owned/swift/THIRD_PARTY_NOTICES.md) · [DeepSeek third-party notices](owned/deepseek/THIRD_PARTY_NOTICES.md)
