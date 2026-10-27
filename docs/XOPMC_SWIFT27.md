# Swift 1.5 Qwen3.8 27B Heretic

This dense 27B model has its own retained native Strata adaptation. It is distinct from the much larger MoE Flash Next family and from its mixed NVFP4 implementation.

## Model artifacts

Own gentle ARA/Heretic BF16 was produced from the ukisai Swift 27B checkpoint. The Heretic BF16 was retained. The native Q4_K_M artifact is 18,201,634,848 bytes plus a 931,146,528-byte BF16 vision projection. Native MTP matrices/norms are retained, and protected ARA writers remain in their intended precision. The main Q4 model was not requantized during speed optimization.

MLX affine 4-bit / group 64 was also prepared with native MTP assets and a nested compatible DFlash2 head. Strict CPU loading was checked; no Mac inference speed is inferred from that check. The unmodified [base model](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) is a provenance link, not a download of our private Heretic artifacts.

## Selected runtime and measurements

The original v19 / API r9 deployment selected async Q4 KV prefetch, pairwarp GDN and compact recurrence/rollback state, fused FFN prompt work, Sage prefill and separately capped Main/draft ubatches. Main Q4_K_M stays unchanged; recorded KV is Q4_0. The target verifies DFlash proposals. Stock DFlash2 Q4 remains selected at 128K; Marathon R32 Q4 at 256K. Native MTP was tested slower in the recorded 256K comparison and was not selected. Draft and native-MTP comparison geometries were different, so that comparison is not a bitwise-parity proof.

| Actual new input / capacity | New-prompt tok/s | Generation tok/s | Repeat-generation tok/s |
| --- | ---: | ---: | ---: |
| 129,024 / 131,072 | 2351.820 | 122.284 | 121.758 |
| 260,096 / 262,144 | 1923.998 | 108.275 | 108.184 |

Fresh-input rows have zero cached prompt tokens and 1,792 generated tokens. The repeat-generation column is a separate repeated-prompt observation, not another new-prompt throughput result. One run per selected configuration; no broad quality or portability claim follows.

## Per-device deployment

Independent original 128K replicas were tested sequentially on historical GPU 1–6, then the original gateway's per-GPU ownership/TTL/API lifecycle was checked separately. Sequential speed numbers are not measurements of six simultaneous inference rates. A GPU replacement affects only that device's instance; default idle TTL was 7,200 seconds. Historical device identities are not a portable allowlist.

## Source publication status

The original runtime and launchers remain in the canonical local model section. Their source closure has not been imported into the public `owned/` CMake selector. The original release contained historical fixed-device gates that require the same current-index admission port used elsewhere before portable publication. Do not reuse its old identity whitelist or relabel the FlashNext provider as a 27B implementation. No 27B CMake/launch target is claimed here.

[Sanitized selected and replica receipts](data/owned-swift27-results.json) contain actual metrics and hashes of the original saved receipts. Model weights, user prompts, credentials and private machine paths are not included.
