# Swift classic source reconstruction

This source-only snapshot reconstructs the selected Swift TwoGPU V454 native closure with the V458 KV-prefetch policy. It is staging material for the unified Strata fork, not a new performance-qualified release.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_CUDA_HOST_COMPILER=clang++-18 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --target strata_swift_classic -j 8
```

Clang, mold and a compatible NVIDIA CUDA toolkit are required. The historical target is SM89/CUDA13.4; exact per-translation-unit flags and archive order are preserved in generated CMake. All linked implementation objects are rebuilt from source; no precompiled local binaries or model weights are included. Header changes invalidate object rebuilds.

`closure.json` records193 selected object/archive-member occurrences and171 unique object recipes; each recipe association was located using exact original object byte identity and historical compile commands. This establishes source provenance, not equality of a fresh build. `compile-source-map.json` contains portable flags; paths are relative to this source snapshot. LICENSE and copied source notices remain with their source roots.

## Remaining release checks

The extraction itself performed no compilation. Root subsequently completed a full source-only Clang18+mold/CUDA13.4 build and actual filled512k plus fresh16k model parity; see QUALIFICATION.json. Historical source-content pins still need comparison, with the remaining historical unsealed inputs documented; the completed512k model check does not establish every context. Selected V301/V311 gates were ported in staging to this-launch ordered STRATA_RUN_GPU_UUIDS/CUDA_VISIBLE_DEVICES plus actual CUDA identity/count checks. SM89 and original numerical/version guards remain; see device-policy-port.json. Other unselected historical examples are not release entrypoints. Missing old BF16 include directories are retained in provenance; they were already absent from the current source workspace and may be unused include search entries. The clean build must determine whether a real dependency is missing.

The generated `_private` exporter is local research tooling and is excluded from publication. Do not copy a runtime credential, model payload, private build directory or machine-specific launcher to this package.
