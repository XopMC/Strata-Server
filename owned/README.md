# Owned native providers

The source-only providers preserve the numerical implementation and headers belonging to each measured backend. They are built through one CMake entry with explicit family/topology flags; upstream Strata remains the default project.

```sh
cmake -S owned -B build-owned -G Ninja \
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_CUDA_HOST_COMPILER=clang++-18 -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSTRATA_OWNED_FAMILY=swift -DSTRATA_OWNED_GPU_COUNT=2
cmake --build build-owned --target strata_owned_swift -j8
```

Current source integration: Swift two-card V454 native closure, with the separately qualified V458 `STRATA_KV_PREFETCH=1` launch policy. Additional Swift1/3/4 and DeepSeek2 source closures are included with explicit family/count/context selectors; their newly reconstructed binaries remain unqualified for deployment. The imported launch registry refuses these profiles until native model qualification. Unsupported choices fail instead of selecting a different numerical backend. A common source build entry is not yet the completed runtime/profile consolidation. No old precompiled server archives are used to satisfy a source target.

The historical target is Linux/x86-64, AdaSM89 and CUDA13.4, using Clang+mold and per-source CPU flags. HIP/SYCL/Metal upstream support does not imply that these new native providers have been ported or tested there. Model weights remain external. See each provider's source manifest for provenance and remaining qualification, [measurements](../docs/XOPMC_RESULTS.md) and [P2P policy](../docs/XOPMC_P2P.md).

## Select a native profile

The source selector takes current NVIDIA system indices. It resolves a transient ordered visibility snapshot at each launch; no UUID is saved as an allowlist. The initial integrated profile admits two cards and512k/1M. Other selections fail explicitly while their source ports are pending.

```sh
python3 owned/launch.py --profile swift-nvfp4-two --gpus 0,1 --context 512k \
  --pack /models/swift/native-pack-experts-v1 \
  --common /models/swift/common.gguf --mtp /models/swift/mtp-production-q4 \
  --expert-profile /models/swift/expert-profile.bin --plan
```

Remove `--plan` only after building and providing matching model assets. `--native-server` selects the engine's stdin protocol; it is not an HTTP listener or a claim that the app/server integration is complete. Existing upstream entrypoints remain unchanged.

An optional `--shared-expert-arena` must refer to a valid sealed arena from a live compatible owner. There is no automatic fallback reload on invalid shared state. Without that argument the ordinary one-process model loader is used. CPU affinity and NUMA policy are deployment choices; the published throughput rows used32 workers, main CPU127 and interleaved two-socket memory. Do not compare a differently scheduled run as an identical profile.

CPU-only selector checks: `python3 owned/test_launch.py`. P2P transaction checks can be built independently from `owned/peer-policy-tests`; they make no CUDA calls. Full source/model qualification remains separate.

The native GEN protocol must explicitly include `ckpt=0 pin=0 pipe=0 aux_cpus=0` in each request. Its historical checkpoint default is not admitted by this source profile. The selected HTTP delivery adapter supplied these fields; the low-level `--native-server` flag alone is not that adapter. Do not feed a default upstream checkpoint request to it.

## NUMA changes and reboot

The owned selector defaults to interleaving across freshly discovered online, allowed memory-bearing nodes. It does not persist `0,1` as the whole machine. `--memory-policy inherit` is available for an explicitly configured parent policy. CPU pinning is separate. The shared helper refuses invalid topology rather than applying an old list.

After a BIOS NPS change, verify the new CPU/node/GPU locality and requalify expert scheduling. NPS4 can expose eight memory domains on this two-socket server; it is not equivalent to the measured two-node setup. A `/dev/shm` arena and its ready seal do not survive reboot. Validate model mounts, prove the old owner is absent, populate one new-boot arena once, then start one CPU keeper and sequential native children. No automatic old sweep, driver modification or second RAM bank is started by this source checkout.
