"""Read-only K14 process-counter evidence; stdlib only, no DSO loading.

The 20-word layout is the sealed pair-indexer-engine-v1-r2 C ABI. Maxima,
live-owner gauge and latest epoch are kept separate from additive counters.
"""
import ctypes as C
import json

FIELDS = ("schema", "owner_wrappers", "active_owners", "disabled_initial_position",
          "disabled_context_batch", "source_publications", "joined_split_calls",
          "declined_calls", "gpu0_query_rows", "gpu1_query_rows", "logical_new_key_bytes",
          "logical_helper_qw_bytes", "logical_returned_top_bytes", "maximum_produced_key_extent",
          "initialized_epochs", "successful_all_stream_drains", "poisoned_owners",
          "live_initialized_owners", "maximum_explicit_helper_bytes", "latest_initialized_epoch")
ADDITIVE = tuple(range(1, 13)) + (14, 15, 16)
MAXIMA = (13, 18)


def require_policy(policy, settings=None):
    policy.assert_current(settings)
    expected = {"DS1017_PAIR_INDEXER_PREFILL_V1":"0", "DS41_INDEXER_S8_R1":"1",
                "DS1017_PAIR_PREFILL_V1":"0", "DS41_PREFILL_GU_HIDDEN_V41":"0",
                "DS1017_PAIR_STAGE_COPY_V4":"0", "DS1017_PAIR_PREFILL_LOOKAHEAD_V5":"0",
                "DS41_PF_PROFILE":"1", "DS41_NATIVE_PAIR_CACHE_MODE":"off",
                "DS41_NATIVE_PAIR_CACHE_GIB":"0"}
    actual = dict(policy.environment)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError("K14 counter evidence requires immutable numerical policy: " + key)


def words(values):
    values = tuple(values)
    if len(values) != 20 or any(type(v) is not int or not 0 <= v < 1 << 64 for v in values) or values[0] != 1:
        raise ValueError("K14 counter export requires schema1 and exactly20 uint64 words")
    return values


def read_counters(library, policy):
    require_policy(policy)
    try:
        function = library.ds1017_pair_indexer_info_v1
    except AttributeError as error:
        raise RuntimeError("Actual selected DSO lacks ds1017_pair_indexer_info_v1") from error
    function.argtypes = [C.POINTER(C.c_uint64), C.c_int]
    function.restype = C.c_int
    output = (C.c_uint64 * 20)()
    if function(output, 20) != 0:
        raise RuntimeError("Actual ds1017_pair_indexer_info_v1 counter read failed")
    require_policy(policy)
    return words(output)


def expected_geometry(prompt_tokens, context, shape):
    if type(context) is not int or context not in (262144, 524288):
        raise ValueError("K14 evidence requires selected256/512k context")
    if type(prompt_tokens) is not int or not 512 <= prompt_tokens <= context or not isinstance(shape, dict):
        raise ValueError("K14 evidence requires an actual at-least512-row ordinary prompt")
    passes, chunk, batch = (shape.get(k) for k in ("passes", "chunk_tokens", "attention_batch"))
    if (any(type(v) is not int for v in (passes, chunk, batch)) or not 1 <= chunk <= min(context, 262144)
            or not 512 <= batch <= 2048 or passes != (prompt_tokens + chunk - 1) // chunk):
        raise ValueError("K14 evidence differs from actual bounded prefill geometry")
    calls = local_rows = helper_rows = publications = small = mirrored_extent = 0
    for start in range(0, prompt_tokens, chunk):
        end = min(start + chunk, prompt_tokens)
        for position in range(start, end, batch):
            count = min(batch, end - position)
            publications += int((position + count) // 2 > position // 2)
            if count < 512:
                small += 1
                continue
            cut = (count // 512) * 256
            calls += 1; local_rows += cut; helper_rows += count - cut
            mirrored_extent = (position + count) // 2
    if not calls:
        raise ValueError("K14 evidence requires actual helper split activity")
    return dict(owner_wrappers=1, active_owners=1, initialized_epochs=1,
                disabled_initial_position=0, disabled_context_batch=0, poisoned_owners=0,
                successful_all_stream_drains=2, source_publications=3 * publications,
                joined_split_calls=3 * calls, gpu0_query_rows=3 * local_rows,
                gpu1_query_rows=3 * helper_rows, logical_new_key_bytes=3 * mirrored_extent * 128 * 2,
                logical_helper_qw_bytes=3 * helper_rows * (32 * 128 * 2 + 32 * 2),
                logical_returned_top_bytes=3 * helper_rows * 512 * 4,
                small_tail_declines_minimum=3 * small, produced_extent=prompt_tokens // 2,
                source_layers=[2,8,14], actual_prefill_shape=dict(shape), prompt_tokens=prompt_tokens)


def validate_prefill_counters(before, after, *, prompt_tokens, context, shape, policy, settings):
    require_policy(policy, settings)
    if settings.context != context:
        raise ValueError("K14 proof context differs from immutable runtime settings")
    before, after = words(before), words(after)
    expected = expected_geometry(prompt_tokens, context, shape)
    delta = {FIELDS[i]:after[i] - before[i] for i in ADDITIVE}
    failures = [FIELDS[i] + " decreased" for i in ADDITIVE + MAXIMA + (19,) if after[i] < before[i]]
    for key, value in delta.items():
        if key in expected and value != expected[key]:
            failures.append(key + " differs from actual geometry")
    if delta["declined_calls"] < expected["small_tail_declines_minimum"]:
        failures.append("small tails were not declined")
    if before[16] or after[16] or before[17] or after[17]:
        failures.append("poisoned or live owner at prefill boundary")
    if after[19] != before[19] + 1:
        failures.append("latest epoch must advance exactly once")
    if after[13] != max(before[13], expected["produced_extent"]) or after[13] > context // 2 + 1:
        failures.append("produced-key maximum differs from actual published extent")
    if not 0 < after[18] < 512 << 20:
        failures.append("explicit helper allocation outside512MiB bound")
    report = dict(schema="ds1017-pair-indexer-prefill-proof/1", abi="ds1017_pair_indexer_info_v1/20",
                  boundary="immediately before ordinary prefill / after synchronous prefill and shape",
                  before=dict(zip(FIELDS,before)), after=dict(zip(FIELDS,after)), delta=delta,
                  maxima_before={FIELDS[i]:before[i] for i in MAXIMA},
                  maxima_after={FIELDS[i]:after[i] for i in MAXIMA},
                  latest_epoch_before=before[19], latest_epoch_after=after[19], expected=expected,
                  policy_environment=dict(policy.environment), counter_validation_passed=not failures,
                  numerical_policy="original K14 leaves; ratio2 complete-query split; ratio1/S8 unchanged",
                  numerical_parity_admitted=False,
                  transfer_units="logical payload bytes; each key/QW/top payload uses two DMA legs; not measured PCIe traffic",
                  failures=failures)
    if failures:
        raise ValueError("K14 prefill counter evidence refused: " + json.dumps(report,sort_keys=True,separators=(',',':')))
    require_policy(policy, settings)
    return report
