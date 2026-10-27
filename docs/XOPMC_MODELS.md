# XopMC model downloads

The public model collection is [huggingface.co/XopMC513](https://huggingface.co/XopMC513). The two repositories below were verified public on28October2026. Weights are distributed separately from Strata-Server source.

| Model | Published variants | Matching backend family |
| --- | --- | --- |
| [Swift1.5 Qwen3.8 Flash Next Heretic](https://huggingface.co/XopMC513/Swift-1.5-Qwen3.8-Flash-Next-Heretic) |Own gentle-Heretic original BF16, Q4_K_M GGUF, custom mixedNVFP4 package, matching assets|Swift/Qwen4exp|
| [DeepSeekV4.1 Flash1017 Heretic](https://huggingface.co/XopMC513/DeepSeek-V4.1-Flash-1017-Heretic) |Own gentle-Heretic in original tensor precision and custom mixed quantization|DeepSeekV4.1 native provider|

The model cards describe the adaptation, tensor policies, available files, licenses and evaluation scope. Their original base-model licenses and notices still apply; the source repository's code license is not a replacement model license.

## Swift formats

Q4_K_M is a GGUF model; mixedNVFP4 is a complete custom native package. Download all required expert/common/tokenizer/vision/MTP components specified by the package manifest. A common-weights GGUF component alone is not the complete NVFP4 model. The current public card lists111.06GB for the complete native package and119.35GB for the Q4_K_M GGUF plus separate vision projector.

Ada does not have native Blackwell FP4 matrix instructions. The owned backend keeps NVFP4 packed in memory and uses the compatible qualified CPU/GPU kernels; the format name alone does not establish arithmetic speed. Use the model's matching production MTP assets with target verification, rather than substituting an unrelated draft head by filename.

## Source integration status

Original qualified backend results and the rebuilt source providers are identified separately in [XOPMC_RESULTS.md](XOPMC_RESULTS.md). A model download being complete does not prove every context, topology or API integration in a new source port. Unsupported profiles fail explicitly; there is no automatic replacement with an older slow numerical path.

The upstream desktop installer remains available through the original Strata project, but it does not automatically contain these fork-specific native providers. Build the owned provider from this checkout and follow its profile configuration. See [owned source build](../owned/README.md), [hardware](XOPMC_HARDWARE.md) and [upstream Strata](https://github.com/Niko1221/Strata).
