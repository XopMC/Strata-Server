# Additional selected Swift source profiles

These source-only CMake manifests import current public one-GPU selections, selected three-GPU native comparison V431, and the experimental four-GPU V421 implementation into the same source tree as the two-GPU provider. Every implementation is rebuilt from owned source files; there are no paths to privately compiled engines.

| Profile | Physical object occurrences | Status |
|---|---:|---|
| swift1-256k |136|Current public packet/item15 source selection|
| swift1-512k / swift1-1024k |140 each|Current public long R60c source selection; same native program, different runtime policy|
| swift3-v431 |162|Selected three-dense native comparison; latest V444 measured256k/512k|
| swift4-v421-experimental |171|Experimental; not a performance winner or production promotion|

The profile fragments have distinct archive/output paths and reuse identical object targets by original objectSHA. Root must integrate the fragments into the common CMake/entrypoint, then run clean source builds and numerical/model checks. No such checks were run by the import. Static source association is not fresh binary equivalence or a quality benchmark.

`profiles/source-provenance.json` records every added file, existing-hash collision preserved without overwrite, excluded uncompiled translation units, and literal include reachability. Physical duplicate ar member names remain distinct. Headers/licenses are retained. Existing Root device/P2P ports were preserved; legacy profile admission still needs review in the common entrypoint. No existing source file was overwritten.

Portable launch settings are reported separately to Root in the publication research report; they are copied from actual selected one-GPU launch configs and V444 three-GPU receipts, not invented from the two-GPU variant. Device identities and private paths are absent from that report.
