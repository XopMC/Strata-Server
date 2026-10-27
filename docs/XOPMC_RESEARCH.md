# Upstream and adjacent-project research

Audit: 28 October 2026. Findings below separate incorporated source work from experiments and future candidates. External kernel results are not local model throughput.

## Strata and related CUDA engines

The review covered the complete upstream range ending at Strata0.2.7 / commit `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d`:367 commits and534 changed paths, rather than only a truncated web comparison. The imported upstream tree remains the default project. Owned numerical providers are explicitly selected from `owned/`.

| Source or mechanism | Disposition in this fork |
| --- | --- |
| [Strata](https://github.com/Niko1221/Strata), current DeepSeek/CPU expert/prefill/MTP work | Reviewed; owned closures retain their matching numerical ABI and measured source implementations. |
| Layer-major prefill | Candidate, not enabled for custom NVFP4: its current format admission and accumulation order need separate work. |
| Interleaved Q6 prefetch leaf | Prepared as a separate candidate; not credited to the release without numerical and model qualification. |
| Copy-event ownership and release-wait deduplication | Useful architecture candidate with epoch/lifetime constraints; no unmeasured speed claim. |
| [Project Maya](https://github.com/mw00/project-maya) | Exact small-rank communication and separate speculative-state groups are candidates. Reduced-precision wire modes are not silently substituted. |
| [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/commit/899e9ed3319a776416fc7042c6a9ece188a67b64) | Whole-output-row work partitioning and compact activation exchange are useful directions. A packed-KV scheme is lossless only relative to its own original quantization, not arbitrary BF16. |
| Route skipping and shortened attention windows | Not selected: owned comparison profiles keep all experts and full attention. |

Published native source changes include shared expert-bank ownership, required process-local P2P admission, current-index device discovery, explicit topology/source selection and the qualified two-card closure. See [source status](../owned/README.md), [results](XOPMC_RESULTS.md) and [P2P contract](XOPMC_P2P.md).

## Uzu, reviewed after the fork push

[Uzu](https://github.com/trymirai/uzu) was reviewed at commit [`0d60981a899c79fc1c29ab3ca1c5aeff937affff`](https://github.com/trymirai/uzu/tree/0d60981a899c79fc1c29ab3ca1c5aeff937affff). The inspected tree is primarily Rust/Metal for Apple hardware. It does not supply a ready CUDA NVFP4 or Ada multi-GPU P2P implementation for this server.

Concrete future candidates are shape- and device-specific dispatch; independent GEMV output-row, input and reduction tile choices; explicit speculative-tree budgets; trained target-feature tap identities; GDN ancestry-aware verification; committing recurrent state only along the accepted serial path; and distinct constant, scratch and sparse-pool lifetimes. These are design references, not imported CUDA kernels. Metal fast approximations and different SIMD reduction order need separate numerical admission. Uzu's U4 storage must not be relabeled NVFP4.

No Uzu code, benchmark or performance claim is part of the qualified release. Candidate plans were recorded in the model backlog; they do not alter current launchers.
