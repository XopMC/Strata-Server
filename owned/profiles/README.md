# Owned runtime profiles

Profiles bind a numerical source provider to its recorded model settings. They do not download model assets, select historical device identities, or substitute another backend. `native_profiles` must match the SHA-stamped provider built with `STRATA_SWIFT_PROFILE`.

| Runtime profile | CMake numerical profile | Context | Public qualification |
|---|---|---|---|
| `swift-nvfp4-one-256k` | `swift1-256k` |256k|Source-ready; original production selection; rebuilt model pending|
| `swift-nvfp4-one-512k` | `swift1-512k` |512k|Source-ready; original production selection; rebuilt model pending|
| `swift-nvfp4-one-1m` | `swift1-1024k` |1M|Source-ready; original production selection; rebuilt model pending|
| `swift-nvfp4-two` | `swift2-v454` |512k/1M|See its own context qualification; source-rebuilt512k qualified|
| `swift-nvfp4-three-256k` | `swift3-v431` |256k|Source-ready; selected original native comparison; rebuilt model pending|
| `swift-nvfp4-three-512k` | `swift3-v431` |512k|Source-ready; selected original native comparison; rebuilt model pending|
| `swift-nvfp4-four-experimental-256k` | `swift4-v421-experimental` |256k|Experimental original completed control; not production winner|

`source-ready-registry.json` is an index, not a launchable profile. New one/three/four-GPU profiles have `launch_enabled:false`: copying source and historical runtime settings does not qualify the rebuilt model. Experimental four-GPU execution must additionally require explicit launcher experimental opt-in after its missing runtime-resource inputs are supported. Three-GPU1M and four-GPU512k/1M are deliberately absent; no supported configuration has been inferred from another topology.

The native argument vector and environment are derived from the exact selected configurations. In particular, one-GPU256k and longer contexts use different source closures and different environment flags; three-GPU256k/512k have different reserve/min-active settings. They cannot be merged by changing context alone. A source-only build stamp is necessary but insufficient to claim the recorded original performance.

All asset placeholders require explicit caller inputs. `pack`, `common`, `expert_profile` and `mtp` describe existing compatible model assets. Library search paths must resolve the freshly built runtime/system dependencies; no privately compiled engine is supplied. Three/four-GPU shared arenas must be compatible existing sealed banks: avoid a second populated RAM copy, and never append the arena argument twice. Historical CPU/NUMA settings are retained as evidence; launch admission must exclude reserved uploader CPUs64–67 and honor the current authorized scope.

The four-GPU original protocol is `GEN {max_new} spec_min_p=0.5 pcie_frac=0 {encoded_prompt_ids}` without an `aux_cpus` override. Its exact original control used CLI speculation6, suffix3, KV prefetch0 and16384-token prefill. Both arena epoch environment fields require fresh descriptor values. Its reference-resource path and first-logits output are separate inputs; a generic directory cannot replace both. The profile remains disabled until a launcher explicitly validates these inputs and experimental admission.

The launcher must reject disabled profiles before execution, validate exact native-profile stamps, expand environment placeholders, apply the current CPU/NUMA policy and enforce experimental opt-in. The registry does not itself implement those checks. No current GPU UUID or PCI address is persisted; resolve current indices at each launch and validate only that launch's transient identity snapshot. Native peer transport remains mandatory for communicating visible devices, without a host/old-path fallback.
