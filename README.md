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

Our checkpoints are published under **[XopMC513 on Hugging Face](https://huggingface.co/XopMC513)**. Open a model below for its formats, backend, measurements and build commands. Weights are separate downloads; this repository contains source, profiles and documentation.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Swift 1.5 Qwen3.8 Flash Next Heretic</b></summary>

**[Download on Hugging Face →](https://huggingface.co/XopMC513/Swift-1.5-Qwen3.8-Flash-Next-Heretic)**

Our gentle-Heretic Swift is available as original-precision weights, **Q4_K_M**, and a **custom mixed NVFP4** package. The native work covers CPU experts, one shared RAM bank, GPU expert residency, prefill, attention and target-verified MTP.

### Swift performance

These are our local, single-request measurements. New-prompt speed counts newly encoded input. Long rows use genuinely filled contexts; output is 1,024 real generated tokens.

| Backend | GPUs | New input / allocated context | Generation | New prompt |
| --- | ---: | ---: | ---: | ---: |
| Mixed NVFP4, **unified source rebuild** | 2 | 520,000 / 524,288 | **192.34 tok/s** | **9,265 tok/s** |
| Mixed NVFP4, original qualified implementation | 2 | 1,040,000 / 1,048,576 | **187.64 tok/s** | **7,753 tok/s** |
| Mixed NVFP4, original prefill-oriented profile | 3 | 260,000 / 262,144 | **177.89 tok/s** | **11,457 tok/s** |
| Mixed NVFP4, original prefill-oriented profile | 3 | 520,000 / 524,288 | **185.38 tok/s** | **12,321 tok/s** |

The three-GPU prefill and two-GPU generation results belong to different profiles. Their best numbers are not combined into an imaginary configuration. More cards did not automatically improve generation; four-GPU work remains experimental.

**Small fill in a long allocation:** the unified two-GPU build measured **225.05 generation tok/s** and **3,298 new-prompt tok/s** at **16,384 / 524,288** tokens. This was a fresh short request in the same process, not a filled 512K result.

Historical single-GPU medians were **173.26 / 181.79 / 175.29 generation tok/s** and **5,297 / 4,752 / 3,833 new-prompt tok/s** at 256K / 512K / 1M. Those used a different adaptive profile and are not a controlled scaling comparison.

### Swift source profiles

| GPUs | Context choices | Current qualification |
| ---: | --- | --- |
| 1 | 256K / 512K / 1M | Context-specific source imported; rebuilt-model qualification pending |
| 2 | 512K / 1M | Source build and filled-512K parity checked; rebuilt 1M qualification separate |
| 3 | 256K / 512K | Prefill-oriented source imported; rebuilt-model qualification pending |
| 4 | Experimental | Requires `STRATA_OWNED_EXPERIMENTAL=ON`; not a production winner |

The final two-GPU source build matched the original control's **initial logits, prefill geometry and all 1,024 output token IDs**, on both 520,000-token and 16,384-token inputs. Expert cache maps stayed the same; the shared bank was reused with **zero expert payload bytes reloaded**. Shutdown drained captured resource pools.

The long-fill control measured 192.22 generation / 9,244 new-prompt tok/s; the rebuild measured 192.34 / 9,265. This establishes reconstruction parity on these tests; the small timing difference is not promoted as an optimization gain.

### Build Swift

After cloning the repository, build the selected two-GPU provider with the qualified toolchain:

```sh
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

`--gpus` selects current NVIDIA system indices. `--plan` loads no CUDA or weights. `--native-server` selects the native stdin protocol; it is not an HTTP listener. A valid `--shared-expert-arena` reuses a sealed expert bank, not request KV or a pre-reboot bank.

[Swift formats and assets](docs/XOPMC_MODELS.md) · [Full measurements](docs/XOPMC_RESULTS.md) · [Source qualification](owned/swift/QUALIFICATION.json) · [Profile registry](owned/profiles/README.md)

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeek V4.1 Flash 1017 Heretic</b></summary>

**[Download on Hugging Face →](https://huggingface.co/XopMC513/DeepSeek-V4.1-Flash-1017-Heretic)**

Our gentle-Heretic DeepSeek is available as an **original-precision checkpoint** and **mixed NVFP4-based quantization**. The separate native backend covers compressed attention, CPU/GPU experts and target-verified speculative decoding.

Original precision preserves the checkpoint's tensor formats, including FP8/MXFP4 where present. It is not an all-BF16 model. The custom mixed package requires its matching loader; a file extension alone does not establish compatible arithmetic.

### DeepSeek performance

Our original two-GPU native API profile, one request with 512 real output tokens:

| Backend | GPUs | New input / allocated context | Generation | New prompt |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek 1017 mixed, original native API | 2 | 261,568 / 262,144 | **54.69 tok/s** | **4,936 tok/s** |
| DeepSeek 1017 mixed, original native API | 2 | 523,712 / 524,288 | **56.64 tok/s** | **5,590 tok/s** |

These are filled-context measurements of the original native implementation. They are not new measurements of the reconstructed source provider.

### DeepSeek source profile

| GPUs | Context choices | Current qualification |
| ---: | --- | --- |
| 2 | 256K / 512K | Source checks passed; clean build and common serving-factory qualification pending |

The imported provider retains matching numerical headers, original per-source arithmetic flags and explicit current-device locality checks. Its unqualified common serving factory remains disabled.

### Build DeepSeek

After cloning the repository, choose the DeepSeek family and one context:

```sh
cmake -S owned -B build-deepseek -G Ninja \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.4 \
  -DSTRATA_OWNED_FAMILY=deepseek \
  -DSTRATA_OWNED_GPU_COUNT=2 \
  -DSTRATA_OWNED_CONTEXT=262144

cmake --build build-deepseek --target strata_deepseek_classic -j8
```

Use `524288` for the 512K source variant. This builds native source; it does not enable the unqualified serving factory or claim parity of a newly built binary.

[DeepSeek formats and assets](docs/XOPMC_MODELS.md) · [Full measurements](docs/XOPMC_RESULTS.md) · [Native source status](owned/deepseek/STATUS.md) · [Native C ABI](owned/deepseek/C_ABI.md)

</details>

## Benchmarks

All model-specific tables are inside the model blocks above. Measurements use the dual-EPYC / RTX 4090 server below, **new input tokens**, one request and the stated actual fill. Individual long-context runs and historical single-GPU medians are identified separately. Component bandwidth is not model throughput, and these results are not guarantees for another machine.

[Full results, settings and limits →](docs/XOPMC_RESULTS.md)

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

The owned providers are opt-in. Start with the shared checkout:

```sh
git clone https://github.com/XopMC/Strata-Server.git
cd Strata-Server
```

Open the matching model block above for its build commands, GPU/context choices and qualification. Native source profiles use **Clang 18 + mold**, CUDA 13.4 and SM89. Model assets remain separate. Building the upstream project alone does not select an owned native provider.

The shared peer-policy API handles up to seven selected compatible devices. This does not mean seven-GPU model execution has been implemented or measured. Unqualified launch profiles are disabled; unsupported choices fail explicitly. Integration of every owned provider with a common HTTP factory remains separate from the retained upstream app and APIs.

After a BIOS NPS change, CPU placement and memory admission must be checked again. The selector discovers current allowed memory nodes instead of assuming `0,1` is the whole machine. Shared RAM seals and request state do not survive reboot.

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
