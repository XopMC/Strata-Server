# DeepSeek classic source reconstruction

Imported into XopMC/Strata-Server under `owned/deepseek` from the sealed staging
whitelist. `provenance/STAGING_SOURCE_READY.json` retains the original source
seal; the local `SOURCE_READY.json` covers this import and its explicit disabled
factory guards and the current-device NUMA locality derivative. Full license texts and attributions are in `LICENSES` and
`THIRD_PARTY_NOTICES.md`. See `C_ABI.md` for exact target/context commands and
the authoritative C declarations. Root build files were not modified.

Native558 source closure is reconstructed for contexts 262144 and 524288.
It includes actual factory563 CPU418/Task489 and cold565's lossless output16
packer. All 72 unique selected object/shared-library hashes have a matching
historical compiler/link command. The export contains sources and notices only.
There are no copied objects, archives, shared libraries, model files or secrets.

**This is not yet a complete runnable publication.** The old Python delivery
requires private historical content/receipt bindings. Fourteen older Python
modules with fixed GPU identities were refused rather than publishing those
bindings or guessing how to replace them. `lifecycle-manifest.json` names each
one. Factory563 source and the actual cold565 lifecycle implementation are
included, with literal source paths rebound to this source tree. No historical
seal, ready bit, machine profile, benchmark prompt or private receipt is shipped.
They must be bound to the clean build through the common publication lifecycle.
Actual factory563 `create`, cold565 `launch_plan`, `serve_child` and `execute`
refuse immediately behind `PUBLICATION_FACTORY_ENABLED = False`. The CMake
serving-factory option is OFF and enabling it is refused. No environment switch
pretends this source import is a qualified serving backend.

## Build the native source island

The current-device NUMA derivative has not been compiled or run. No CUDA call,
topology query or model run was performed by the source exporter. The standalone build uses installed CUDA 13.4,
Clang 18, llvm-ar and mold, plus the system C++/pthread/dl/math/libmvec/OpenMP
dependencies used by the selected code. The CPU profile retains its original
znver4/physical-core assumptions; a different CPU is a new admission/proof task.

```sh
cmake -S . -B build-256 -DSTRATA_DS_CONTEXT=262144
cmake --build build-256 --target strata_deepseek_classic

cmake -S . -B build-512 -DSTRATA_DS_CONTEXT=524288
cmake --build build-512 --target strata_deepseek_classic
```

The output is `<build>/<context>/libds1017_pair_runtime_v1.so`; newly built
support DSOs live under `<build>/support`. Select one context per live GPU
child. Both main engines intentionally retain the original C ABI and SONAME;
loading both into one interposable namespace is not admitted. The architecture
classes use their own exact matching include islands. Do not compile them
against the fresh fork's similarly named classes or link old private objects.

`build-manifest.json` preserves per-TU options, ordered include directories,
archive members and the original engine link order, including repeated archive
entries. Host compiler/toolkit paths are rebound to installed tools. Dependency
output flags are removed. Support DSOs receive basename SONAMEs and relative
ORIGIN RPATHs to remove runtime dependencies on the build machine. These are
build/lifetime changes; no numerical source operator or class/C-ABI layout is
rewritten. Three diagnostic-only C exports also receive a narrow visibility
port: BF16-many probe, WOA-many probe and V320 headless-resource component now
compare a fresh transient visibility snapshot with runtime UUID order, retaining
their original visible counts 1/1/2 and owner ordinal 0. Static UUID/PCI bindings
are removed. No production numerical leaf or native-P2P admission is changed;
this helper is not a replacement for required all-pairs startup admission.
`diagnostic-admission-delta.json` records source hashes and the exact scope;
removed historical identity literals are redacted from its diff.
The same ledger records one additional production **device guard** port in
the byte-only V141 inverse: a saved two-UUID byte array becomes this launch's
transient snapshot. Its existing `cuDeviceGetUuid` call, owner ordinal, stream,
capture, tag139, pointer/range and overlap checks remain; the permutation kernel
is unchanged. This is not a numerical operator or peer-admission replacement.

Both selected contexts also replace the two saved PCI NUMA lookups used by the
host stage-copy pools. At pool creation the calling thread's current CUDA device
is queried with `cudaGetDevice`, its PCI address with `cudaDeviceGetPCIBusId`,
and that address's current sysfs `numa_node` is read. Missing or invalid topology
throws before worker creation. No device is selected by this lookup and no node
is guessed. Existing explicit CPU-list/locality checks, worker pinning, joined
copies, streams and class layouts remain unchanged. The automatic pool retains
its original CPU-ID bound; the explicit serial mode retains its CPU0..3 policy.
After a topology change, an incompatible CPU list therefore refuses the copy
pool and must be admitted afresh. `current-numa-topology-delta.json` records the
four source hashes and scope proofs. This is source qualification only, and the
serving factory remains disabled pending the real clean-build/lifecycle port.

CUDA host links use Clang+mold. These compiler/rebinding changes
still require actual clean-build numerical parity qualification by Root.

CPU539 specifically retains `-fno-fast-math -ffp-contract=off -frounding-math
-march=znver4`. No blanket fast-math, architecture substitution, new quantizer,
model weight change or host transport fallback is added.

`export_closure.py --workspace /path/to/canonical-workspace` reconstructs the
same staging closure. It reads metadata and existing object/archive bytes for
provenance; it does not run compilers, import a runtime, load a DSO or read model
payloads. `verify_source.py` is a finite CPU-only manifest/source check.
The recovery script is retained as provenance: rerun it only in a fresh staging
directory. Running it over this publication import would invalidate its import
seal and disabled-factory guard delta. It is not part of the native CMake build.

## Provenance and remaining integration

Entry evidence is `research/pass-upload-async-native-v558/ROOT_RECIPE.json`
plus actual `state/pass-upload-async-native-root-build-v558.json` in the
DeepSeek1017 TwoGPU workspace. Actual558 receipt commands take precedence over
prepared recipes. `recovery-index.json` records every member/object SHA and
chosen command evidence with historical workspace paths replaced by a marker.
No basename-only source substitution is used. Every selected archive has unique
member names, checked before `ar p`; each occurrence is recorded as 1. An archive
with duplicate names is rejected and needs occurrence-aware recovery.

Important recovered overlays include V175 FP8LT/scaled epilogue, compact V252
GPU experts, actual558 context engines, CPU539, direct531 FP8, K5 V503/V504,
context-specific K14 V334, fixed469 owner/resources, whole501, literal53/276,
the original R6 FP8 and K2/K3/K7/K8/K10/K12/K13/K15 leaves. The selected native
DSO is not the fresh upstream `src/dsv41` implementation.

The minimum remaining gates are concrete:

1. Root clean source build/link and C-ABI idle checks for each context. Then
   same selected checkpoint/input/geometry/logits/output/MTP rollback comparison.
   Historical DSO hashes describe old artifacts, not the new build's identity.
2. Replace private validation-history bindings with a fresh common publication
   binder. Keep original numerical/counter/lease checks; do not make stale source
   SHA checks pass by ignoring them. Exported Python is explicitly not ready.
3. Resolve the fourteen refused old fixed-identity modules against actual
   selected adapter overrides. Do not use a generic basename match or rewrite a
   saved UUID into another saved UUID. Dynamic loader call sites are enumerated.
4. Keep one CPU-only bank owner, current-index resource-set leases, fresh CUDA
   child per context, native4 MTP and original Engine-close/drain behavior.
   Original563 `create(store, context)` returns `run(body,cancel,emit)`,
   `configuration()` and `close()`; it admits full-chat prefill, not KV append.
   Cold565 remains pending independent actual cold-start qualification.
5. Common required-native-peer admission must run in that selected GPU child
   before model allocation/capture. Keep direct resident transport and positive
   producer/consumer credits. Admission alone does not port old host transport.
6. External model/auxiliary paths, state paths, bind/auth and source bindings
   belong to explicit install configuration. No current machine profile is a
   public default. `STRATA_MODEL_DIR` is required for transformed external model
   literals; no asset download is attempted by this source packet.

`source-manifest.json` records original/exported hashes and include-only path
rewrites. Original Strata/exllamav3 licenses are retained. The 387 required
CUTLASS/CUTE headers retain their embedded NVIDIA BSD-3-Clause notices; they
are source headers, not a copied environment. A complete publication license,
secret and source review remains Root's final gate. No benchmark achievement,
quality improvement, cold-start result or portable binary parity is claimed.
