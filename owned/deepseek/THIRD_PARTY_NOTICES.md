# DeepSeek source island: licenses and attribution

This directory imports the selected native558/factory563/cold565 source closure.
The native source base is Strata `native-ds41` revision
`c6cadcc9692345af72b163d955142f6e6d012398`. The publication import does not
replace the copyright or license of any underlying source. Existing notices
remain in their original files. Full license texts are supplied under `LICENSES`.

| Material in this source closure | Attribution | Full license |
| --- | --- | --- |
| Strata native-ds41 and legacy Strata source trees | Copyright 2026 Niko1221 and the Strata contributors | `LICENSES/Strata-MIT.txt` |
| `native-ds41/third_party/exllamav3_moe` and `exllamav3_gpu` leaves | Copyright 2025 Turboderp | `LICENSES/ExLlamaV3-MIT.txt` |
| 387 required CUTLASS/CUTE headers | NVIDIA Corporation and affiliates; individual years/notices remain in each file | `LICENSES/NVIDIA-CUTLASS-CUTE-BSD-3-Clause.txt` |
| `tools/native_fp8_scaled_epilogue_v170.cu` | Contributors to the vLLM project; native raw-pointer adaptation 2026, as stated in its retained header | `LICENSES/vLLM-Apache-2.0.txt` |

The vLLM adaptation identifies v0.8.5
`csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh` in its original source notice.
The complete Apache-2.0 license was obtained from that release's primary
repository. A root `NOTICE` file was not present at the inspected v0.8.5 path;
no upstream NOTICE was invented. The source's copyright, adaptation notice and
CUTLASS attribution are retained.

`tpurtell/ds41rt` at revision
`3067d0684a0b572d88a17d38ca3f7d10599619b4` appears in the surrounding historical
architecture/configuration references. No file from that repository was found
in the selected publication whitelist. Its full MIT text, copyright 2026 T.J.
Purtell, is nevertheless provided in `LICENSES/TPurtell-ds41rt-MIT.txt` as an
explicit reference attribution. This is not a claim that all DeepSeek code was
imported from ds41rt, or that its license applies to other contributors' work.

The historical design kit and separately referenced MiaAI/Engram/other external
runtimes are not vendored by this whitelist. In particular, their presence in
nearby research does not relicense any future imported code under Strata's MIT
license. No model weights, model license grant, dependency binaries or CUDA SDK
source are bundled here. Installed CUDA/driver and runtime package licenses
remain separate.

CUTLASS/CUTE headers retain their historical include island under
`sources/DeepSeek V4.1/work/deepseek41/.venv-reference/.../cutlass/include`.
Only listed text headers are included, not a Python environment. The embedded
BSD notices are kept, and the full BSD text extracted from the shipped
`cutlass/cutlass.h` notice is additionally available as a standalone license.

`LICENSES/SOURCE_PROVENANCE.json` records the pinned primary license URLs and
download hashes. `source-manifest.json` and the preserved staging source seal
record exact source provenance. The imported inference numerical sources and
their class/C ABI layouts are unchanged. Publication changes add a disabled
factory gate and license/build documentation; the already recorded four
visibility-admission ports remain as imported.
