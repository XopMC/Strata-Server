# XopMC native backend results

This fork preserves Strata's app, Terminal and API. The owned native providers collect measured Swift and DeepSeek changes behind explicit family and topology choices. They keep architecture-specific numerical state separate: an MoE loader, speculative rollback and attention cache cannot safely share a C++ class layout merely because two models use experts.

## What the measurements describe

These are measurements of the original qualified local source closures on two AMD EPYC9754 CPUs and RTX4090 GPUs. They are not new measurements of this reconstructed checkout. Each row states actual newly encoded input and allocated capacity; an allocated long context alone is not a filled-context result. Generation and new-prompt processing are separate. Prompt caching is off for these rows. The Swift two-card rows generated1024 varied tokens, with verified Q4 MTP and int8 KV. All experts and full attention are retained; no expert skipping or approximate attention is credited.

| Model / configuration | Actual new input | Allocated context | Generation, tokens/s | New prompt, tokens/s | Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Own Swift MixedNVFP4, two GPUs |520000|524288|191.245|9239.205|V454+V458, one run|
| Same process, next fresh short request |16384|524288|223.342|3281.197|one run, not a long fill|
| Own Swift MixedNVFP4, two GPUs |1040000|1048576|187.635|7752.522|V454+V458, one run|
| Same process, next fresh short request |16384|1048576|222.044|3219.113|one run, not a long fill|
| Own Swift MixedNVFP4, three GPUs, prefill role |260000|262144|177.886|11456.722|V431/V444, one run|
| Same three-card profile |520000|524288|185.376|12321.136|one run|
| Own DeepSeek1017 mixed, two GPUs |261568|262144|54.694|4936.491|V558/V563 API,512 outputs|
| Same DeepSeek profile |523712|524288|56.643|5590.103|512 outputs|

The three-card prefill profile and the two-card generation profile are different executions. Their best numbers cannot be combined into a hypothetical single row. Four-card experiments are retained as research; there is no selected four-card runtime. A source-only rebuild still needs matched validation before inheriting these numbers.

Historical single-card Swift medians of three runs were173.263/181.786/175.285 generation and5297.420/4751.744/3833.364 new-prompt tokens/s at256k/512k/1M. Those used a different adaptive profile from later fixed-role cross-topology experiments; they do not establish scaling by themselves.

## Clean-source reconstruction check

The two-card Swift provider was rebuilt entirely from source with Clang18, mold and CUDA13.4. On actual520000/524288 it produced192.340 generation and9265.182 new-prompt tokens/s; the matched original control was192.217/9243.804. Following fresh16384/524288 it produced225.055/3297.575 versus224.537/3296.447. These small differences are not a new speed claim. All1024 output IDs, initial logits and prefill geometry matched; cache maps were unchanged, the existing CPU arena was reused with0 expert payload reads, and native shutdown retired captured pools. See [machine-readable qualification](../owned/swift/QUALIFICATION.json). Source-rebuilt1M and other providers remain separately unqualified.

## Changes retained in the selected Swift two-card closure

- Native NVFP4 CPU arithmetic and exact GPU expert leaves; the storage format does not imply native FP4 Tensor Core instructions on Ada.
- Shared read-only expert arena with explicit ready/seal ownership, so successive selected GPU workers can attach without reloading the full expert payload. A CUDA process is started fresh; CUDA state is never forked from a model owner.
- Physical expert ownership, bounded adaptation and retained-cache loans for prompt staging. One RAM weight copy does not imply one GPU copy or one shared request KV state.
- GPU routing, paired gate/up work, protected residual arithmetic and original aggregation order.
- Pooled virtual-memory QSA workspace, qualified current-length selection schedules and direct peer transport. A small prompt remains a modern native schedule, rather than entering a historical host path.
- Existing KV staging-pool prefix prefetch. In the matched1M run this improved new-prompt throughput7553.046→7752.522(+2.641%), while generation187.402→187.635. All1024 output IDs and initial logits matched.
- Matching target-verified Q4 MTP. Draft speed is not output speed: the target verifies emissions and owns rollback.

Conditional CPU-credit variants, forced512k peer selection and other exact experiments that slowed whole-model generation are not production defaults. The experimental MTP classifier row split and new upstream IL_PF leaf are being tested separately; neither is credited to the source release yet.

## Models and formats

- [Swift1.5 Flash Next Heretic](https://huggingface.co/XopMC513/Swift-1.5-Qwen3.8-Flash-Next-Heretic): own original-precision Heretic, Q4_K_M, and custom mixedNVFP4 package.
- [DeepSeekV4.1 Flash1017 Heretic](https://huggingface.co/XopMC513/DeepSeek-V4.1-Flash-1017-Heretic): own original-precision Heretic and mixed quantization.

Weights are separate downloads. This code repository contains no checkpoints, credentials, private API keys or machine model paths. Custom native packages require their matching loader; a renamed extension does not convert one format into another.

## Reproduction and limits

Use the source build in [owned](../owned/README.md) and preserve the exact family, topology, context, KV, MTP, CPU affinity and expert budgets. Record GPU identity as a fresh launch snapshot, never a stored allowlist. Native P2P is required for owned multi-card profiles; missing peer access is a startup error. It is not permission to alter a user's driver or power settings.

Source provenance and clean-build status accompany each provider. Numerical parity is tested independently from speed. A successful kernel test or a large external Tensor Core peak is not an end-to-end throughput result. These measurements are specific to the listed hardware/profile and are not guarantees of broader quality or speed.

## Complete Swift Flash Next 1–4 GPU inventory

This table lists original historical profiles and the tested unified source rebuild. Single-GPU values are medians of three runs. The remaining numerical rows are individual completed runs. Different profiles/epochs are not a controlled hardware scaling curve. Missing rows are explicitly unmeasured in the audited receipt set.

| GPUs | New input / allocated context | Generation | New prompt | Recorded profile |
| ---: | ---: | ---: | ---: | --- |
| **1** | 260,000 / 262,144 | **173.26 tok/s** | **5,297 tok/s** | Historical adaptive, median of 3 |
| **1** | 520,000 / 524,288 | **181.79 tok/s** | **4,752 tok/s** | Historical adaptive, median of 3 |
| **1** | 1,040,000 / 1,048,576 | **175.29 tok/s** | **3,833 tok/s** | Historical adaptive, median of 3 |
| **2** | 260,000 / 262,144 | **209.04 tok/s** | **9,005 tok/s** | Original shared-bank profile |
| **2** | 520,000 / 524,288 | **192.34 tok/s** | **9,265 tok/s** | **Unified source rebuild** |
| **2** | 1,040,000 / 1,048,576 | **187.64 tok/s** | **7,753 tok/s** | Original selected V454/V458 |
| **3** | 260,000 / 262,144 | **177.89 tok/s** | **11,457 tok/s** | Original prefill profile V444 |
| **3** | 520,000 / 524,288 | **185.38 tok/s** | **12,321 tok/s** | Original prefill profile V444 |
| **3** | 1,040,000 / 1,048,576 | **173.90 tok/s** | **10,778 tok/s** | Historical three-dense V55 |
| **4** | 260,000 / 262,144 | **169.18 tok/s** | **5,939 tok/s** | Experimental shared-bank control |
| 4 | — / 524,288 | — | — | No completed result in the audited set |
| 4 | — / 1,048,576 | — | — | No completed result in the audited set |

Alternative measured profiles, kept separate:

| GPUs | New input / allocated context | Generation | New prompt | Alternative profile |
| ---: | ---: | ---: | ---: | --- |
| 3 | 260,000 / 262,144 | 205.04 tok/s | 8,725 tok/s | Original generation-oriented profile |
| 4 | 260,000 / 262,144 | 169.38 tok/s | 5,997 tok/s | Later four-GPU V28 experiment, not selected |

[Sanitized evidence and TTFT](data/owned-swift-1-4-gpu-results.json).

## Swift 1.5 Qwen3.8 27B

The separate original Q4_K_M backend measured 122.284 / 2,351.820 generation / new-prompt tok/s at 129,024 / 131,072 tokens, and 108.275 / 1,923.998 tok/s at 260,096 / 262,144 tokens. Both generated 1,792 tokens, using Q4_0 KV and target-verified DFlash2. Six historical 128K replicas were tested sequentially as independent instances. The 27B runtime is not yet imported into the public `owned/` selector. [Backend details](XOPMC_SWIFT27.md) · [Sanitized receipts](data/owned-swift27-results.json).
