# Required native P2P in XOPMC profiles

Owned multi-GPU profiles require native CUDA peer access. A missing direction
stops startup. There is no automatic host staging or legacy-runtime retry.
CPU expert computation and the original RAM weight arena remain valid parts of
CPU+GPU profiles; they are separate from inter-GPU tensor transport.

The driver capability policy covers all current and future compatible system
cards. A model still selects only its requested current system indices. This
API does not expand that selection, patch a driver, change power settings, reset
a card, or reboot the system. Seven devices is an admission bound, not a claim
that seven installed cards have been tested.

## Callable API

`include/strata/core/required_native_peer.hpp` exposes
`strata::core::require_native_peer(options, receipt, error)`. The implementation
is `src/core/required_native_peer.cpp`; production targets compile it as C++17
with CUDA headers and link CUDA Runtime. This is a CUDA-specific profile API.
It does not make HIP, SYCL or Metal implement native CUDA P2P.

The launcher restricts visibility before the CUDA runtime is initialized. It
resolves the requested **current system indices afresh on every launch**. It
must not use saved UUIDs or PCI identities as a persistent allowlist. A fresh
transient UUID list is useful: numeric `CUDA_VISIBLE_DEVICES` with
`CUDA_DEVICE_ORDER=PCI_BUS_ID` must not be assumed to use the system's current
NVML/nvidia-smi index order.

Choose one binding contract:

- `FreshUuidSnapshot`: supply `admitted_devices`, `current_system_indices` and
  full `transient_cuda_uuids` in the intended CUDA ordinal order. The API checks
  the visible count and each runtime UUID. The caller provides the freshness
  and current-system-index provenance of that transient snapshot.
- `CallerValidatedVisibility`: supply `admitted_devices` after the launch layer
  has already validated selection and ordering. Current system indices are
  optional receipt diagnostics. Leave `transient_cuda_uuids` empty. The API
  checks the exact visible count but does not claim to independently verify
  system-index ordering in this mode.

The historical launcher variable `STRATA_RUN_GPU_UUIDS` may hold a fresh
transient visibility snapshot. The API neither requires a variable spelling nor
changes environment variables. It never treats those UUIDs as permanent policy.

Example at the CUDA-child startup hook:

```cpp
strata::core::RequiredNativePeerOptions options;
options.admitted_devices = launch.selected_count;
options.binding = strata::core::NativePeerBinding::FreshUuidSnapshot;
options.current_system_indices = launch.fresh_system_indices;
options.transient_cuda_uuids = launch.fresh_cuda_order_uuids;
options.before_model_allocation_and_capture = true;
strata::core::RequiredNativePeerReceipt receipt;
std::string error;
if (!strata::core::require_native_peer(options, receipt, error)) {
    // Report error and terminate this GPU child. No alternate transport.
    throw std::runtime_error(error);
}
```

This hook must run in the GPU child **after fork**, once during serialized
startup, before model device allocation and capture on any stream. The RAM
keeper/download/conversion parent remains CPU-only. The explicit options flag
is the caller's assertion about that phase: querying the default stream cannot
prove that no other stream has captured, nor detect every earlier allocation.
Do not call from a CUDA host callback or during concurrent context mutation.

The API examines only the selected visible cohort. Every selected device must
support unified addressing. It checks all directed capabilities before creating
any mapping, enables all `N*(N-1)` directions, and restores the entry device.
For one selected device the pair count is zero, but phase, visibility, identity
and capture checks still apply. Success prints `STRATA_P2P_DEFAULT` with counts
and `host_fallback=0`.

Existing mappings remain enabled. On admission failure the API attempts to undo
only mappings created by that call, in reverse order, and restores the entry
device. `receipt.admitted` is false on every failure. A failed cleanup also sets
`cleanup_complete=false`; terminate the child. Successful mappings live until
context teardown. A receipt destructor does not disable them. After context
reset/replacement an old receipt supplies no admission credit.

## Transport and lifetime remain explicit

Admission is required but is not a transport or numerical proof. Each profile
must route GPU-resident tensors through direct peer transport. On the qualified
local stack, captured transport uses UVA
`cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, owning_stream)`.
`cudaMemcpyPeerAsync` worked outside capture but was not capture-compatible in
the recorded qualification. Admission performs neither kind of copy.

For every transfer retain source and destination allocations through the last
consumer, with the correct stream owner. A fresh producer event/dependency must
precede a peer read; the consumer's positive completion must precede buffer
reuse, adaptation, loan return or destruction. Host submission, an old event,
or a successful peer enable is not a completion credit. Graphs must retain all
borrowed resources until their last launch is drained and they are destroyed.
On errors poison the request, drain the owning work, and refuse acceptance.
Do not add host mailbox transport or a legacy fallback to handle failure.

Preserve original expert bytes, numerical formats, route order, ordered
aggregation, attention/KV semantics and MTP checks. Enabling peer access does
not justify changing any of them. Existing mapped-host expert paths are not
converted by adding this startup call; their actual transport needs a separate
port and independent proof before a profile claims compliance.

## Policy provenance and validation

This transaction is derived from the canonical workspace's
`Shared/work/cuda-peer-default/POLICY.md` and qualified `peer_matrix.hpp`, read
on 2026-10-28:

- Policy SHA256: `6edf0db3d66fffab131f1b3127694853aff2b5146b56f87d455265cd136af09c`.
- Helper SHA256: `c5f342114abc1c72fe8d47b4961d723d513f62bbfadd9a6aaa844513dbb0ec9d`.

The reusable API preserves the all-pairs transaction and failure rollback. It
adds explicit caller phase/binding contracts, optional runtime-order validation,
unified-addressing checks, a receipt, and CPU-isolated fault tests. It does not
copy a machine-specific identity allowlist into this repository.

Historical local qualification covered three selected cards, six directions,
36 data checks and two mapping cycles. Later read-only capability observations
are not a payload/capture proof for future hardware. This new module has not
been CUDA-built or GPU-tested by its source author. No new bandwidth or model
performance result is claimed.

`tests/native_peer_policy.cpp` exercises this same transaction with a fake
runtime: counts 1..7, fresh ordering, malformed metadata, active capture,
missing UVA/directions, repeated admission, every successful-path runtime call
failing or throwing, reverse rollback, cleanup refusal, and inherited mappings.
It performs no CUDA initialization. The source author's 2026-10-28 Clang 18 +
mold run passed 1,930 CPU checks. A CPU-only command is:

```sh
clang++-18 -std=c++17 -Wall -Wextra -Werror -fuse-ld=mold -DSTRATA_NATIVE_PEER_CPU_ONLY \
  -Iinclude src/core/required_native_peer.cpp tests/native_peer_policy.cpp \
  -o /tmp/strata-native-peer-policy && /tmp/strata-native-peer-policy
```

`STRATA_NATIVE_PEER_CPU_ONLY` is only for that isolated test target. With it the
production entry always refuses admission, rather than pretending CUDA worked.
The internal `native_peer_detail::Runtime` seam is for policy tests, not a
selectable profile transport. The owning integration must add the TU/test target
and the startup hook; those build/main files are outside this source packet.
Actual CUDA compilation, per-launch device binding, pairwise payload/capture
checks, stream/resource lifetime and whole-model parity remain integration gates.
