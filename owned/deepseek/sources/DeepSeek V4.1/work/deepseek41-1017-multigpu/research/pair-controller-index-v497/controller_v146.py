#!/usr/bin/env python3
"""Root-only foreground controller for one shared CPU bank and paired GPU children.

Importing this module is CPU-only. The parent imports the canonical OneBankStore,
preloads it once, and then forks exactly one fresh pair-runtime child at a time.
The runtime adapter and Torch are imported only inside that child.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import select
import signal
import sys
import subprocess
import time
import traceback
import uuid
from maintenance_v146 import Manager,assert_child_binding
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = PROJECT.parent
WORK = PROJECT / "work"
MULTI = WORK / "deepseek41-1017-multigpu"
OUTPUT_BASE = PROJECT / "outputs" / "deepseek41-1017-multigpu" / "pair-controller-v214"
OWNER_ROOT = WORK / "deepseek41-1017-nvfp4" / "research" / "old-complete-runtime-1017-adapter-v1"
OWNER_SOURCE = OWNER_ROOT / "one_bank_store.py"
P_SOURCE = WORK / "deepseek41-1017-nvfp4" / "delivery" / "single-gpu-frozen-v1" / "source-snapshot" / "one_bank_store.py"
PAIR_ADAPTER_ROOT = MULTI / "research" / "pair-runtime-adapter-v2"
PAIR_BUILD_DEFAULT = MULTI / "state" / "pair-runtime-build-v1-r2.json"
PAIR_PREFILL_BUILD_DEFAULT = MULTI / "state" / "pair-prefill-engine-build-v2.json"
PAIR_PREFILL_COMPONENT_DEFAULT = MULTI / "state" / "pair-prefill-engine-component-v2.json"

from index_devices_v497 import resolve_current_pair
SELECTED_INDICES = (0, 1)
CURRENT_PAIR_V497 = resolve_current_pair()
# Per-launch snapshot only: neither value is a persistent hardware allowlist.
GPU0, GPU1 = (row["uuid"] for row in CURRENT_PAIR_V497)
PAIR_MASK = GPU0 + "," + GPU1
CONTEXTS = (262144, 524288)
CACHE_SIZES = (0, 8, 16, 20)
EXPECTED_OWNER_SHA = "d4794c02ac19442f821579f4a1f1d049fe574bbbd6cf398d5b6e013feb4acedc"
DEFAULT_PAIR_ADAPTER_SEAL_SHA256 = "afe9266ecb1c4e7339afaa31561dbe1e7c233f9abcf3249eaf0945bc55669d59"
GIB = 1 << 30
MAX_JSON_BYTES = 64 << 20
MAX_RECEIPT_BYTES = 8 << 20
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ControllerError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


def _regular_no_symlink(path: Path, *, limit: int) -> os.stat_result:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ControllerError(f"Cannot stat required file: {path}") from exc
    _need(not path.is_symlink() and path.is_file(), f"Expected a regular nonsymlink file: {path}")
    _need(before.st_size <= limit, f"File exceeds its bound: {path}")
    return before


def _read_pinned_bytes(path: Path, expected_sha256: str, *, limit: int) -> bytes:
    _need(bool(SHA_RE.fullmatch(expected_sha256)), "Invalid SHA256 binding")
    before = _regular_no_symlink(path, limit=limit)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        now = os.fstat(fd)
        _need((now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns) ==
              (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns),
              f"File changed while opening: {path}")
        chunks = []
        remaining = now.st_size
        while remaining:
            chunk = os.read(fd, min(8 << 20, remaining))
            _need(bool(chunk), f"Short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        _need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
              (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns),
              f"File changed while reading: {path}")
    finally:
        os.close(fd)
    _need(hashlib.sha256(raw).hexdigest() == expected_sha256, f"SHA256 mismatch: {path}")
    return raw


def _read_json(path: Path, expected_sha256: str, *, limit: int = MAX_RECEIPT_BYTES):
    raw = _read_pinned_bytes(path, expected_sha256, limit=limit)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError(f"Invalid JSON receipt: {path}") from exc
    _need(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _sha_file(path: Path, *, max_bytes: int) -> str:
    before = _regular_no_symlink(path, limit=max_bytes)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        now = os.fstat(fd)
        _need((now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns) ==
              (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns),
              f"File changed while opening: {path}")
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            h.update(block)
        after = os.fstat(fd)
        _need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
              (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns),
              f"File changed while hashing: {path}")
    finally:
        os.close(fd)
    return h.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _need(not path.is_symlink(), f"Refusing symlink output: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    os.replace(tmp, path)
    dirfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _canonical_json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _validate_body(body: dict, job: dict) -> None:
    _need(isinstance(body, dict), f"Body for {job['id']} is not a JSON object")
    maximum = body.get("max_completion_tokens", body.get("max_tokens", 512))
    _need(type(maximum) is int and maximum == job["expected_completion_tokens"],
          f"Body completion length differs for {job['id']}")
    _need(body.get("temperature", 0) == 0 and body.get("repetition_penalty", 1) == 1,
          f"The matched greedy/native4 test requires temperature=0 and repetition_penalty=1: {job['id']}")
    _need(isinstance(body.get("messages"), list) and body["messages"], f"Missing explicit messages: {job['id']}")


def _cpu_group_sum(job: dict) -> str:
    value = job.get("CPU_group_sum", "0")
    _need(type(value) is str and value in ("0", "1"), "CPU_group_sum must be literal string0 or1")
    return value


def _job_arm(job: dict) -> tuple:
    return (job["cache_mode"], job["cache_gib"], job["pair_prefill"], job["mtp"], _cpu_group_sum(job))


def _factory_cpu_options(adapter_root: Path, job: dict) -> dict:
    """Legacy adapters receive their unchanged signature; V11 receives explicit mode."""
    match = re.fullmatch(r"pair-runtime-adapter-v([2-9]|[1-9][0-9]+)", adapter_root.name)
    _need(match is not None, "Unsupported CPU policy adapter packet name")
    mode = _cpu_group_sum(job)
    if int(match.group(1)) >= 11:
        return {"CPU_group_sum": mode}
    _need(mode == "0", "CPU group-sum ON requires a versioned V11-or-newer adapter")
    return {}


def _expected_cpu_projection(adapter_root: Path, job: dict) -> dict | None:
    """Called inside the admitted fresh child; this stdlib resolver loads no DSO."""
    if not _factory_cpu_options(adapter_root, job):
        return None
    version = int(adapter_root.name.rsplit("v", 1)[1])
    name = f"cpu_projection_binding_v{version}"
    binding_module = __import__(name)
    _need(Path(binding_module.__file__).resolve() == adapter_root / "source" / f"{name}.py",
          "CPU binding module did not come from the explicitly selected adapter")
    binding = binding_module.resolve_cpu_projection()
    _need(binding.mode == _cpu_group_sum(job), "Resolved CPU projection mode differs from requested job")
    return binding.record()


CPU_COMPONENT_SCOPES_V174=frozenset(("synthetic-correctness-only", "synthetic148-and-retained18-matrices108-component-correctness; full-model pending"))

def _validate_expected_cpu_projection_v174(expected,job):
    if expected is None:
        _need(_cpu_group_sum(job)=="0","Legacy binding cannot assert CPU group-sum ON")
        return
    _need(isinstance(expected,dict) and expected.get("mode")==_cpu_group_sum(job),"Expected CPU projection mode differs")
    _need(isinstance(expected.get("path"),str) and Path(expected["path"]).is_absolute() and isinstance(expected.get("sha256"),str) and SHA_RE.fullmatch(expected["sha256"]),"Expected CPU projection requires actual path/SHA")
    _need(expected.get("proof_scope") in CPU_COMPONENT_SCOPES_V174 and expected.get("model_pending") is True,"CPU component scope must remain explicit known component-only and model_pending")
    _need(all(expected.get(key,False) is False for key in ("quality_admitted","model_speed_admitted","performance_admitted")),"CPU component cannot assert model quality/speed admission")


def _validate_cpu_result(runtime: dict, job: dict, expected: dict | None) -> dict | None:
    mode = _cpu_group_sum(job)
    actual = runtime.get("CPU_projection")
    if expected is None:
        _need(mode == "0" and actual is None,
              "Legacy CPU0 mode cannot accept an unvalidated CPU projection binding")
        return None
    _validate_expected_cpu_projection_v174(expected,job)
    _need(isinstance(actual, dict) and actual == expected,
          "Runtime CPU_projection differs from selected adapter's actual pinned path/SHA/mode/proof")
    _need(runtime.get("CPU_group_sum") == mode
          and runtime.get("policy_environment", {}).get("DS1017_CPU_GROUP_SUM_V2") == mode,
          "Runtime CPU group-sum mode differs from explicit requested job")
    return dict(actual)


def _validate_request_object(req: dict):
    _need(req.get("schema") == "deepseek1017-pair-controller-workload/1", "Unsupported workload schema")
    _need(ID_RE.fullmatch(str(req.get("run_id", ""))) is not None, "Invalid run_id")
    owner = req.get("owner")
    runtime = req.get("runtime")
    jobs = req.get("jobs")
    _need(isinstance(owner, dict) and isinstance(runtime, dict) and isinstance(jobs, list) and jobs,
          "Workload requires explicit owner/runtime bindings and at least one job")
    for name in ("bundle_root", "manifest_sha256", "source_revision", "packer_path", "packer_sha256",
                 "packer_proof_path", "packer_proof_sha256", "integrity_receipt_path", "integrity_receipt_sha256"):
        _need(name in owner, f"Missing owner binding: {name}")
    for name in ("pair_build_receipt_path", "pair_build_receipt_sha256",
                 "expected_adapter_seal_sha256", "adapter_wait_seconds"):
        _need(name in runtime, f"Missing runtime binding: {name}")
    _need(bool(SHA_RE.fullmatch(owner["manifest_sha256"])), "Invalid bundle manifest SHA")
    _need(bool(SHA_RE.fullmatch(owner["packer_sha256"])), "Invalid packer SHA")
    _need(type(runtime["adapter_wait_seconds"]) is int and 0 <= runtime["adapter_wait_seconds"] <= 1800,
          "adapter_wait_seconds must be 0..1800")
    if runtime["expected_adapter_seal_sha256"] is not None:
        _need(bool(SHA_RE.fullmatch(runtime["expected_adapter_seal_sha256"])), "Invalid pair adapter seal SHA")
    if "pair_adapter_root" not in runtime:
        _need(runtime["expected_adapter_seal_sha256"] == DEFAULT_PAIR_ADAPTER_SEAL_SHA256,
              "A versioned adapter path is required unless the request pins the frozen v2 adapter")
        runtime["pair_adapter_root"] = str(PAIR_ADAPTER_ROOT)
    adapter_root = Path(runtime["pair_adapter_root"]).absolute()
    _need(adapter_root.parent.resolve() == (MULTI / "research").resolve() and
          re.fullmatch(r"pair-runtime-adapter-v(?:[2-9]|[1-9][0-9]+)", adapter_root.name) is not None and
          adapter_root.is_dir() and not adapter_root.is_symlink(),
          "Pair adapter must be an explicit versioned packet directly under this project research directory")

    ids = set()
    fixtures = {}
    for job in jobs:
        _need(isinstance(job, dict), "Job entry must be an object")
        for name in ("id", "context", "fixture_id", "cache_mode", "cache_gib", "mtp", "body_path",
                     "body_sha256", "expected_prompt_tokens", "expected_prompt_token_ids_sha256",
                     "expected_prefill_shape", "expected_completion_tokens"):
            _need(name in job, f"Job field missing: {name}")
        _need(ID_RE.fullmatch(str(job["id"])) is not None and job["id"] not in ids, "Job IDs must be unique ASCII identifiers")
        ids.add(job["id"])
        job.setdefault("CPU_group_sum", "0")
        _factory_cpu_options(adapter_root, job)
        _need(job["context"] in CONTEXTS, f"Unsupported context: {job['id']}")
        _need(ID_RE.fullmatch(str(job["fixture_id"])) is not None, f"Invalid fixture ID: {job['id']}")
        _need(job["mtp"] in ("off", "native4"), f"Unsupported MTP mode: {job['id']}")
        _need(job.get("pair_prefill") in ("0", "1"), f"pair_prefill must be explicit 0/1: {job['id']}")
        _need(job["expected_completion_tokens"] == 512, f"This controller accepts exactly 512-output checks: {job['id']}")
        _need(type(job["expected_prompt_tokens"]) is int and job["expected_prompt_tokens"] >= 6 and
              job["expected_prompt_tokens"] + job["expected_completion_tokens"] <= job["context"],
              f"Invalid prompt/context geometry: {job['id']}")
        _need(bool(SHA_RE.fullmatch(job["body_sha256"])) and
              bool(SHA_RE.fullmatch(job["expected_prompt_token_ids_sha256"])), f"Invalid input binding: {job['id']}")
        _need(isinstance(job["expected_prefill_shape"], dict), f"Expected actual prefill shape required: {job['id']}")
        _need((job["cache_mode"], job["cache_gib"]) in (("off", 0), ("gpu1", 8), ("gpu1", 16), ("gpu1", 20)),
              f"Only cache0 reference or GPU1 8/16/20 GiB arms are admitted: {job['id']}")
        _need(type(job["expected_prompt_tokens"]) is int and type(job["cache_gib"]) is int, "Integer geometry required")
        job.setdefault("minimum_unique_token_ids", 16)
        job.setdefault("maximum_single_token_run", 64)
        _need(type(job["minimum_unique_token_ids"]) is int and 16 <= job["minimum_unique_token_ids"] <= 512,
              f"Nondegenerate-output unique-token gate must be >=16: {job['id']}")
        _need(type(job["maximum_single_token_run"]) is int and 1 <= job["maximum_single_token_run"] <= 64,
              f"Nondegenerate-output repeat-run gate must be <=64: {job['id']}")
        fixture_key = (job["context"], job["fixture_id"])
        prev = fixtures.setdefault(fixture_key, {"body_sha256": job["body_sha256"], "prompt_tokens": job["expected_prompt_tokens"],
                                                   "prompt_ids_sha256": job["expected_prompt_token_ids_sha256"],
                                                   "prefill_shape": job["expected_prefill_shape"], "arms": set(), "job_order": {}})
        _need((job["body_sha256"], job["expected_prompt_tokens"], job["expected_prompt_token_ids_sha256"],
               job["expected_prefill_shape"]) == (prev["body_sha256"], prev["prompt_tokens"], prev["prompt_ids_sha256"],
                                                  prev["prefill_shape"]),
              f"Matched cache arms must use the same body, tokenization and prefill shape: {fixture_key}")
        arm = _job_arm(job)
        _need(arm not in prev["arms"], f"Duplicate cache arm: {fixture_key} {arm}")
        prev["arms"].add(arm)
    # Resolve stable request order separately. Helper-cache MTP results need an
    # MTP-off run with the same context, fixture, cache and prefill policy first.
    for (context, fixture_id), entry in fixtures.items():
        arms = entry["arms"]
        _need(any(a[0:2] == ("off", 0) and a[2] == "0" for a in arms),
              f"Each context/fixture needs a cache0 reference: {context}/{fixture_id}")
        for index, job in enumerate(jobs):
            if (job["context"], job["fixture_id"]) != (context, fixture_id):
                continue
            if job["cache_mode"] == "gpu1" and job["mtp"] == "native4":
                control = ("gpu1", job["cache_gib"], job["pair_prefill"], "off", job["CPU_group_sum"])
                active = ("gpu1", job["cache_gib"], job["pair_prefill"], "native4", job["CPU_group_sum"])
                _need(control in arms and active in arms, f"Helper-cache MTP needs an off/native4 matched pair: {context}/{fixture_id}/{job['cache_gib']}GiB/prefill{job['pair_prefill']}")
                control_index = next((i for i, other in enumerate(jobs) if (other["context"], other["fixture_id"], *_job_arm(other)) == (context, fixture_id, "gpu1", job["cache_gib"], job["pair_prefill"], "off", job["CPU_group_sum"])), None)
                _need(control_index is not None and control_index < index,
                      f"Helper-cache MTP-off run must complete first: {context}/{fixture_id}/{job['cache_gib']}GiB/prefill{job['pair_prefill']}")
    _need(bool({context for context, _ in fixtures}) and {context for context, _ in fixtures}.issubset(set(CONTEXTS)),
          "Workload must include at least one actual filled 256k or 512k context")
    if any(job["pair_prefill"] == "1" for job in jobs):
        for index, job in enumerate(jobs):
            if job["pair_prefill"] != "1":
                continue
            control = (job["context"], job["fixture_id"], job["cache_mode"], job["cache_gib"], "0", job["mtp"], job["CPU_group_sum"])
            control_index = next((i for i, other in enumerate(jobs) if
                                  (other["context"], other["fixture_id"], *_job_arm(other)) == control), None)
            _need(control_index is not None,
                  f"Prefill-on arm needs a matched prefill-off control with the same MTP/cache: {job['id']}")
            _need(control_index < index, f"Prefill-off control must run before prefill-on: {job['id']}")
    if any(job["cache_mode"] == "gpu1" for job in jobs):
        for job in jobs:
            if job["cache_mode"] == "gpu1" and job["mtp"] == "native4":
                off_control = (job["context"], job["fixture_id"], job["cache_mode"], job["cache_gib"], job["pair_prefill"], "off", job["CPU_group_sum"])
                _need(any((other["context"], other["fixture_id"], *_job_arm(other)) == off_control for other in jobs),
                      f"Helper-cache native4 needs a same-cache/prefill MTP-off control first: {job['id']}")
    if any(job["pair_prefill"] == "1" for job in jobs):
        for key in ("pair_prefill_build_receipt_path", "pair_prefill_build_receipt_sha256",
                    "pair_prefill_component_receipt_path", "pair_prefill_component_receipt_sha256"):
            _need(key in runtime, f"Pair-prefill jobs require an explicit component/DSO receipt binding: {key}")
        for key in ("pair_prefill_build_receipt_sha256", "pair_prefill_component_receipt_sha256"):
            _need(bool(SHA_RE.fullmatch(runtime[key])), f"Invalid pair-prefill SHA binding: {key}")

    return req


def _load_request(path: Path, sha256: str):
    req = _read_json(path, sha256, limit=MAX_JSON_BYTES)
    return _validate_request_object(req)


def _resolve_build_libraries(receipt: dict) -> dict:
    _need(receipt.get("phase") == "PAIR_CABI_COMPILED_AND_LINKED_NOT_EXECUTED", "Pair C ABI build receipt is not compile/link-only")
    out = {}
    for row in receipt.get("contexts", []):
        context = row.get("context")
        _need(context in CONTEXTS and context not in out, "Pair build receipt has duplicate/unsupported contexts")
        path = Path(row["DSO"])
        _need(path.name == "libds1017_pair_runtime_v1.so", "Unexpected base pair runtime DSO name")
        expected = row["DSO_sha256"]
        _need(bool(SHA_RE.fullmatch(expected)) and _sha_file(path, max_bytes=2 << 30) == expected,
              f"Pair runtime DSO hash mismatch: {path}")
        _need(row.get("only_replaced_object", {}).get("original", "").endswith("native-k3-verify-v310/native_engine.cpp.o"),
              "Pair DSO does not derive from the pinned v310 C ABI object replacement")
        out[context] = (path, expected)
    _need(set(out) == set(CONTEXTS), "Pair C ABI receipt must contain exactly the 256k and 512k DSOs")
    return out


def _resolve_pair_prefill_libraries(receipt: dict) -> dict:
    _need(receipt.get("phase") == "PAIR_PREFILL_ENGINE_COMPILED_NOT_EXECUTED",
          "Pair prefill engine build receipt is not compile/link-only")
    out = {}
    for row in receipt.get("contexts", []):
        context = row.get("context")
        _need(context in CONTEXTS and context not in out, "Pair prefill build receipt has duplicate/unsupported contexts")
        path = Path(row["DSO"])
        _need(path.name == "libds1017_pair_prefill_engine_v2.so", "Unexpected pair-prefill DSO name")
        expected = row["DSO_sha256"]
        _need(bool(SHA_RE.fullmatch(expected)) and _sha_file(path, max_bytes=2 << 30) == expected,
              f"Pair prefill DSO hash mismatch: {path}")
        out[context] = (path, expected)
    _need(set(out) == set(CONTEXTS), "Pair prefill receipt must contain exactly the 256k and 512k DSOs")
    return out


def _validate_pair_prefill_component(path: Path, expected_sha256: str) -> dict:
    proof = _read_json(path, expected_sha256)
    _need(proof.get("phase") == "ACTUAL_PAIR_FFN_AND_ORDERED_SCATTER_BIT_PARITY_PASS" and
          proof.get("model_run") is False and
          len(proof.get("runs", [])) == 2 and
          {row.get("name") for row in proof.get("runs", [])} == {"262144-1-full4096", "524288-1-full4096"} and
          all(row.get("exit_code") == 0 for row in proof.get("runs", [])),
          "Pair prefill component needs the actual both-context FFN/scatter parity receipt")
    return proof


def _adapter_modules(adapter_root: Path):
    match = re.fullmatch(r"pair-runtime-adapter-v([2-9]|[1-9][0-9]+)", adapter_root.name)
    _need(match is not None, "Unsupported pair adapter packet name")
    version = match.group(1)
    loader_name = f"pair_runtime_loader_v{version}"
    factory_name = f"pair_factory_v{version}"
    source_dir = adapter_root / "source"
    _need((source_dir / f"{loader_name}.py").is_file() and (source_dir / f"{factory_name}.py").is_file(),
          "Versioned pair adapter lacks the expected loader/factory API")
    return loader_name, factory_name


def _verify_pair_adapter(adapter_root: Path, wait_seconds: int, expected_sha256: str | None, *, cancel_event=None):
    adapter_root = adapter_root.absolute()
    _need(adapter_root.parent.resolve() == (MULTI / "research").resolve() and
          re.fullmatch(r"pair-runtime-adapter-v(?:[2-9]|[1-9][0-9]+)", adapter_root.name) is not None and
          not adapter_root.is_symlink(),
          "Pair adapter path is outside the canonical research directory")
    loader_name, factory_name = _adapter_modules(adapter_root)
    seal_path = adapter_root / "SEALED.json"
    deadline = time.monotonic() + wait_seconds
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise ControllerError("Cancelled while awaiting sealed pair runtime adapter")
        if seal_path.is_file() and not seal_path.is_symlink():
            actual_seal_sha = _sha_file(seal_path, max_bytes=MAX_RECEIPT_BYTES)
            if expected_sha256 is not None:
                _need(actual_seal_sha == expected_sha256, "Pair runtime adapter seal differs from workload binding")
            seal = json.loads(seal_path.read_text())
            members = seal.get("files")
            _need(isinstance(members, list) and members, "Pair adapter seal lacks a bounded member inventory")
            _need(len(members) <= 256, "Pair adapter seal member count exceeds bound")
            for row in members:
                rel = row.get("path", row.get("name"))
                _need(isinstance(rel, str) and rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts,
                      "Unsafe path in pair adapter seal")
                member = adapter_root / rel
                _need(member.resolve().is_relative_to(adapter_root.resolve()), "Pair adapter member escapes packet root")
                _need(member.stat().st_size == row["bytes"] and _sha_file(member, max_bytes=MAX_JSON_BYTES) == row["sha256"],
                      f"Pair adapter seal member mismatch: {rel}")
            loader = adapter_root / "source" / f"{loader_name}.py"
            factory = adapter_root / "source" / f"{factory_name}.py"
            _need(loader.is_file() and not loader.is_symlink() and factory.is_file() and not factory.is_symlink(),
                  "Sealed versioned loader/factory entry missing")
            return actual_seal_sha
        if time.monotonic() >= deadline:
            raise ControllerError("Timed out waiting for the final sealed pair-runtime adapter; no model child was started")
        print(json.dumps({"phase": "WAITING_FOR_SEALED_PAIR_ADAPTER", "path": str(seal_path)}, ensure_ascii=False), flush=True)
        time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))


def _check_parent_environment() -> None:
    _need(sys.dont_write_bytecode, "Launch with PYTHONDONTWRITEBYTECODE=1")
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "The foreground bank owner must keep CUDA_VISIBLE_DEVICES empty")
    _need(os.environ.get("DEEPSEEK41_NUMA_POLICY") == "interleave-both", "Set the admitted NUMA interleave-both policy")
    _need(os.environ.get("DS1017_OLD_COMPLETE_OWNER_V1") == "1", "Explicit new1017 onebank owner opt-in is required")
    _need(not any("libcuda.so" in row for row in Path("/proc/self/maps").read_text().splitlines()),
          "The onebank parent must not have mapped CUDA")
    _need("torch" not in sys.modules, "Torch must be imported only after fork in the GPU child")
    allowed = os.sched_getaffinity(0)
    _need(set(range(256)).issubset(allowed), "Root foreground parent requires the full physical CPU affinity 0..255")


def _require_pair_compute_idle() -> None:
    """Read-only NVML CLI check on the exact admitted pair; parent stays CUDA-empty."""
    for gpu in (GPU0, GPU1):
        proc = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True, timeout=15)
        for line in proc.stdout.splitlines():
            row = [field.strip() for field in line.split(",")]
            if not row or not row[0]:
                continue
            _need(len(row) == 2 and row[1] == gpu,
                  f"NVML returned an unexpected compute-app row for admitted GPU {gpu}")
            raise ControllerError(f"Admitted GPU is not idle: uuid={gpu}, compute_pid={row[0]}")


def _require_pair_idle_recovery_v146():
    _require_pair_compute_idle()
    for gpu in(GPU0,GPU1):
        result=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=uuid,gpu_recovery_action','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15)
        rows=[line.strip()for line in result.stdout.splitlines()if line.strip()]
        _need(len(rows)==1,'exact one health row per admitted UUID required')
        values=[x.strip()for x in rows[0].split(',')]
        _need(len(values)==2 and values[0]==gpu and values[1]in('None','0'),'pair recovery must be actual0; unknown/unsupported refuses')

def _sample_pair_memory() -> dict:
    rows = {}
    for gpu in (GPU0, GPU1):
        proc = subprocess.run(
            ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10)
        parsed = [part.strip() for part in proc.stdout.strip().split(",")]
        _need(len(parsed) == 4 and parsed[0] == gpu, f"NVML memory query returned an unexpected device for {gpu}")
        try:
            total, used, free = map(int, parsed[1:])
        except ValueError as exc:
            raise ControllerError(f"NVML memory query returned malformed numeric fields for {gpu}") from exc
        _need(min(total, used, free) >= 0 and used + free <= total + 32,
              f"NVML memory counters are inconsistent for {gpu}")
        rows[gpu] = {"memory_total_bytes": total * (1 << 20), "memory_used_bytes": used * (1 << 20),
                     "memory_free_bytes": free * (1 << 20)}
    return {"sampled_unix": time.time(), "devices": rows, "source": "read-only nvidia-smi/NVML"}


def _record_pair_memory(path: Path, samples: list, label: str) -> dict:
    sample = _sample_pair_memory()
    sample["label"] = label
    samples.append(sample)
    _atomic_json(path, {"schema": "deepseek1017-pair-controller-gpu-memory/1", "samples": samples})
    print(json.dumps({"phase": "GPU_MEMORY_SAMPLE", **sample}, ensure_ascii=False), flush=True)
    return sample


def _import_owner_module():
    _need(hashlib.sha256(OWNER_SOURCE.read_bytes()).hexdigest() == EXPECTED_OWNER_SHA,
          "Canonical OneBankStore source changed")
    _need(hashlib.sha256(P_SOURCE.read_bytes()).hexdigest() == EXPECTED_OWNER_SHA,
          "Frozen P OneBankStore no longer matches the canonical owner class")
    owner_path = str(OWNER_ROOT)
    if owner_path in sys.path:
        sys.path.remove(owner_path)
    sys.path.insert(0, owner_path)
    import one_bank_store
    _need(Path(one_bank_store.__file__).resolve() == OWNER_SOURCE.resolve(), "Imported OneBankStore came from a copied module")
    _need(hashlib.sha256(Path(one_bank_store.__file__).read_bytes()).hexdigest() == EXPECTED_OWNER_SHA,
          "Imported OneBankStore source hash differs")
    return one_bank_store


def _profile_settings(module, store, job: dict, runtime_dso_path: Path):
    context = job["context"]
    if context == 262144:
        chunk, prefill_batch, expert_batch, margin = 262144, 2048, 4096, 768
        expected_library = "native-engine-speed-combined-cache-stage-copy-r2.so"
    else:
        chunk, prefill_batch, expert_batch, margin = 262144, 1024, 2048, 512
        expected_library = "native-engine-speed-combined-cache-host-topk-r6.so"
    expected_dso = "libds1017_pair_prefill_engine_v2.so" if job["pair_prefill"] == "1" else "libds1017_pair_runtime_v1.so"
    _need(runtime_dso_path.name == expected_dso, f"Unexpected runtime DSO for {context}/prefill{job['pair_prefill']}")
    return module.CppRuntimeSettings(
        model_path=store.source,
        workers=tuple(range(254)),
        coordinator_cpu=255,
        context=context,
        chunk_tokens=chunk,
        prefill_batch=prefill_batch,
        expert_batch=expert_batch,
        packed_ram_gib=448,
        vision_cache_gib=2,
        graph=True,
        decoded_expert_cache=True,
        experimental=True,
        cpu_arithmetic="packed-parallel-post-v66",
        upload_pipeline=True,
        expert_coordinator_cpu=254,
        dense_prefill=True,
        fp8_experts=True,
        expert_cache_gib=job["cache_gib"],
        pair_cache_mode=job["cache_mode"],
        weight_profile=None,
        library_name=expected_library,
        gpu_stage=0,
        dense_fp8=True,
        persistent_experts=False,
        mtp=job["mtp"],
        packed_fp8_weights=True,
        decoder_suffix_prefill=True,
        compact_candidate_prefill=True,
        expert_cache_tail_rows=0,
        prefill_margin_mib=margin,
        mtp_arena_upload=False,
        k3_causal_batch=True,
    )


def _longest_run(values) -> int:
    best = current = 0
    last = None
    for item in values:
        if item == last:
            current += 1
        else:
            last = item
            current = 1
        best = max(best, current)
    return best


def _validate_result(result: dict, job: dict, body_sha: str, *, expected_cpu_projection=None) -> dict:
    _need(result.get("usage", {}).get("prompt_tokens") == job["expected_prompt_tokens"],
          f"Actual prompt length differs: {job['id']}")
    _need(result.get("encoded_prompt_token_ids_sha256") == job["expected_prompt_token_ids_sha256"],
          f"Actual prompt tokenization differs: {job['id']}")
    _need(result.get("actual_prefill_shape") == job["expected_prefill_shape"],
          f"Actual prefill shape differs: {job['id']}")
    ids = result.get("generated_token_ids")
    _need(isinstance(ids, list) and len(ids) == job["expected_completion_tokens"] and
          result.get("usage", {}).get("completion_tokens") == job["expected_completion_tokens"],
          f"Expected exactly 512 generated tokens: {job['id']}")
    _need(all(type(x) is int and 0 <= x < 129280 for x in ids), f"Invalid token ID in output: {job['id']}")
    _need(len(set(ids)) >= job["minimum_unique_token_ids"], f"Output is too repetitive by unique-ID gate: {job['id']}")
    _need(_longest_run(ids) <= job["maximum_single_token_run"], f"Output contains an overlong repeated-token run: {job['id']}")
    mtp = result.get("mtp", {})
    if job["mtp"] == "off":
        _need(mtp.get("mode") == "off" and mtp.get("requested_mode") == "off",
              f"MTP-off control did not stay off: {job['id']}")
    else:
        _need(mtp.get("mode") == "native4" and mtp.get("requested_mode") == "native4",
              f"Native4 MTP did not execute: {job['id']}")
    runtime = result.get("runtime_configuration", {}).get("pair_runtime", {})
    cpu_projection = _validate_cpu_result(runtime, job, expected_cpu_projection)
    _need(runtime.get("mtp") == job["mtp"], f"Runtime-reported MTP mode differs: {job['id']}")
    policy_environment = runtime.get("policy_environment", {})
    _need(policy_environment.get("DS1017_PAIR_PREFILL_V1") == job["pair_prefill"],
          f"Pair-prefill runtime policy differs from the requested arm: {job['id']}")
    cache = result.get("pair_gpu_expert_cache", {})
    _need(cache.get("requested_mode") == job["cache_mode"], f"Reported cache mode differs: {job['id']}")
    devs = cache.get("devices", {})
    _need(devs.get("0", {}).get("uuid") == GPU0 and devs.get("1", {}).get("uuid") == GPU1,
          f"Cache telemetry device UUID mapping differs: {job['id']}")
    _need(devs["0"].get("arena_budget_bytes") == 0 and devs["0"].get("arena_used_bytes") == 0,
          f"GPU0 cache arena must stay empty while native4 MTP is active: {job['id']}")
    if job["cache_mode"] == "off":
        _need(devs["1"].get("arena_budget_bytes") == 0 and devs["1"].get("arena_used_bytes") == 0,
              f"Cache0 control unexpectedly allocated a helper arena: {job['id']}")
    else:
        _need(0 < devs["1"].get("arena_budget_bytes", 0) <= job["cache_gib"] * GIB,
              f"GPU1 helper arena does not match its explicit budget: {job['id']}")
        _need(devs["1"].get("arena_used_bytes", 0) <= devs["1"].get("arena_budget_bytes", 0),
              f"GPU1 helper arena usage exceeds its budget: {job['id']}")
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return {
        "job_id": job["id"],
        "fixture_id": job["fixture_id"],
        "context": job["context"],
        "input_body_sha256": body_sha,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "completion_tokens": result["usage"]["completion_tokens"],
        "prompt_token_ids_sha256": result["encoded_prompt_token_ids_sha256"],
        "output_token_ids_sha256": digest,
        "output_unique_token_ids": len(set(ids)),
        "output_longest_same_token_run": _longest_run(ids),
        "generation_tps": result.get("timings", {}).get("generation_tps"),
        "new_prompt_tps": result.get("timings", {}).get("new_prompt_tps"),
        "actual_prefill_shape": result["actual_prefill_shape"],
        "mtp": mtp,
        "pair_prefill": job["pair_prefill"],
        "CPU_group_sum": _cpu_group_sum(job),
        "CPU_projection": cpu_projection,
        "pair_gpu_expert_cache": cache,
        "GPU_memory": result.get("GPU_memory"),
        "finish_reason": result.get("finish_reason"),
        "quality_admitted": False,
        "cache_arithmetic_quality": "EXPERIMENTAL_UNACCEPTED_GPU_EXPERT_GEMV",
    }


def _child_run(job: dict, body: dict, store, library_path: Path, library_sha256: str,
               adapter_root: Path, cancel_event, outdir: Path, bank_binding=None) -> int:
    runtime_obj = None
    code=1
    try:
        if bank_binding is not None:
            _need(job["pair_prefill"] == "0", "Maintenance mode refuses unqualified pair-prefill DSO")
            assert_child_binding(store,bank_binding)
        os.environ["CUDA_VISIBLE_DEVICES"] = PAIR_MASK
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["DEEPSEEK41_GPU_STAGE"] = "0"
        os.environ["DS41_NATIVE_PAIR_MASK"] = "1"
        os.environ["DS1017_OLD_COMPLETE_RUNTIME_V1"] = "1"
        os.environ["DS41_ATTENTION_WOA_1017_SOURCE_PORT"] = "1"
        os.environ["DEEPSEEK41_NUMA_POLICY"] = "interleave-both"
        os.environ["DS1017_PAIR_PREFILL_V1"] = job["pair_prefill"]
        os.environ["DS41_PREFILL_GU_HIDDEN_V41"] = "0"
        os.environ["DS1017_CPU_GROUP_SUM_V2"] = _cpu_group_sum(job)
        _need("torch" not in sys.modules, "Fresh GPU child imported Torch before pair-mask activation")
        adapter_root = adapter_root.absolute()
        sys.path.insert(0, str(adapter_root / "source"))
        loader_name, factory_name = _adapter_modules(adapter_root)
        loader_module = __import__(loader_name)
        factory_module = __import__(factory_name)
        module = loader_module.load_context_runtime(job["context"])
        settings = _profile_settings(module, store, job, library_path)
        module.validate_cpp_settings(settings)
        expected_cpu_projection = _expected_cpu_projection(adapter_root, job)
        _validate_expected_cpu_projection_v174(expected_cpu_projection,job) # BEFORE factory/model allocations
        runtime_obj = factory_module.create(
            store, context=job["context"], library_path=library_path, library_sha256=library_sha256,
            mtp=job["mtp"], pair_cache_mode=job["cache_mode"], expert_cache_gib=job["cache_gib"],
            packed_ram_gib=448, pair_prefill=job["pair_prefill"], gu_hidden="0",
            **_factory_cpu_options(adapter_root, job))
        final = {}
        def emit(event):
            if event.get("event") == "result":
                final["result"] = event.get("result")
                if isinstance(final["result"],dict):
                    # FIRST durable capture, before any typed/result metadata check.
                    _atomic_json(outdir/"raw-result.json",final["result"])
                    _atomic_json(outdir/"raw-capture.json",dict(schema=174,phase="RAW_RESULT_CAPTURED_UNVALIDATED",child_pid=os.getpid(),job_id=job["id"],runtime_dso_path=str(library_path),runtime_dso_sha256=library_sha256,result_accepted=False,quality_admitted=False))
            return not cancel_event.is_set()
        runtime_obj.run(body, cancel_event, emit)
        _need(isinstance(final.get("result"), dict), "Runtime completed without a final result event")
        _atomic_json(outdir/"raw-result.json",final["result"]) # never removed on validation/close failure
        if bank_binding is not None and bank_binding["layout_tag"]==139:
            _need(final["result"].get("runtime_configuration",{}).get("bank_format_binding_v146")==bank_binding,"Typed paired runtime did not report actual inherited format/epoch")
        summary = _validate_result(final["result"], job, job["body_sha256"],
                                   expected_cpu_projection=expected_cpu_projection)
        if bank_binding is not None:summary["bank_format_binding_v146"]=bank_binding
        summary.update({"runtime_dso_path": str(library_path), "runtime_dso_sha256": library_sha256})
        _atomic_json(outdir / "result.json", final["result"])
        _atomic_json(outdir / "summary.json", summary)
        code=0
    except BaseException as exc:
        _atomic_json(outdir / "failure.json", {"job_id": job["id"], "error": str(exc),
                                                    "traceback": traceback.format_exc(), "child_pid": os.getpid()})
        code=1
    finally:
        try:
            if runtime_obj is not None:runtime_obj.close()
            leases=getattr(store,"_native_leases",-1)
            _need(type(leases) is int and leases==0,"Child store still has native leases after runtime close")
            _atomic_json(outdir/"close-proof.json",dict(schema=174,phase="CHILD_RUNTIME_CLOSED_ZERO_LEASES" if runtime_obj is not None else "CHILD_NO_RUNTIME_CREATED_ZERO_LEASES",runtime_created=runtime_obj is not None,runtime_close_completed=runtime_obj is not None,native_leases=leases,child_pid=os.getpid(),parent_pid=os.getppid(),job_id=job["id"],runtime_dso_path=str(library_path),runtime_dso_sha256=library_sha256,bank_format_binding=bank_binding,result_accepted=False,quality_admitted=False))
        except BaseException as exc:
            code=1
            _atomic_json(outdir/"close-failure.json",dict(schema=174,error=str(exc),traceback=traceback.format_exc(),child_pid=os.getpid(),job_id=job["id"]))
            # Accepted-result filenames must not survive close failure; raw evidence does.
            for name in ("result.json","summary.json"):
                try:(outdir/name).unlink()
                except FileNotFoundError:pass
    return code


def _wait_child(pid: int, cancel_event, *, grace_seconds: int, state_path: Path, job_id: str,
                memory_callback=None):
    started = time.monotonic()
    last_memory_sample = started
    term_sent = False
    kill_sent = False
    kill_at = None
    while True:
        try:
            got, status = os.waitpid(pid, os.WNOHANG)
        except InterruptedError:
            got = 0
        if got == pid:
            return status
        if cancel_event.is_set() and not term_sent:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            term_sent = True
            kill_at = time.monotonic() + grace_seconds
            print(json.dumps({"phase": "CANCELLING_CHILD", "job_id": job_id, "pid": pid}), flush=True)
        if term_sent and kill_at is not None and time.monotonic() >= kill_at and not kill_sent:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            kill_sent = True
            _atomic_json(state_path, {"phase": "CHILD_KILLED_AFTER_GRACE", "job_id": job_id, "pid": pid,
                                      "elapsed_seconds": time.monotonic() - started})
        if memory_callback is not None and time.monotonic() - last_memory_sample >= 30.0:
            memory_callback(f"child-{job_id}-periodic")
            last_memory_sample = time.monotonic()
        time.sleep(0.2)


def _validate_build_receipt(request: dict):
    r = request["runtime"]
    path = Path(r["pair_build_receipt_path"])
    receipt = _read_json(path, r["pair_build_receipt_sha256"])
    libraries = _resolve_build_libraries(receipt)
    prefill_libraries = {}
    component_proof = None
    if any(job["pair_prefill"] == "1" for job in request["jobs"]):
        prefill_receipt = _read_json(Path(r["pair_prefill_build_receipt_path"]), r["pair_prefill_build_receipt_sha256"])
        prefill_libraries = _resolve_pair_prefill_libraries(prefill_receipt)
        component_proof = _validate_pair_prefill_component(
            Path(r["pair_prefill_component_receipt_path"]), r["pair_prefill_component_receipt_sha256"])
    return receipt, libraries, prefill_libraries, component_proof


def _validate_admission_profile_v163(profile: dict) -> None:
    """Only hashes/JSON/source closure; never import candidate Python or DSOs."""
    before = set(sys.modules)
    adapter = Path(profile["adapter_root"])
    _verify_pair_adapter(adapter, 0, profile["adapter_seal_sha256"])
    build = _read_json(Path(profile["pair_build"]["path"]), profile["pair_build"]["sha256"])
    actual = _resolve_build_libraries(build)
    expected = {str(k): {"path": str(p), "sha256": h} for k, (p, h) in actual.items()}
    _need(expected == profile["libraries"], "Profile DSOs differ from actual both-context build receipt")
    _need("torch" not in sys.modules and "torch" not in before, "Profile validation imported Torch")
    _need(not any("libcuda.so" in row for row in Path("/proc/self/maps").read_text().splitlines()), "Profile validation loaded CUDA")


def _preflight_owner_receipts(owner: dict) -> None:
    bundle_root = Path(owner["bundle_root"]).absolute()
    _need(bundle_root.is_dir() and not bundle_root.is_symlink() and bundle_root.resolve() == bundle_root,
          "Bundle root must be the direct canonical model directory, not a symlink")
    packer = Path(owner["packer_path"])
    _need(_sha_file(packer, max_bytes=2 << 30) == owner["packer_sha256"], "CPU output16 packer SHA differs")
    proof = _read_json(Path(owner["packer_proof_path"]), owner["packer_proof_sha256"])
    _need(proof.get("phase") == "ACTUAL_FROZEN_OLD_PACKER_FULL_GEOMETRY_BYTES_AND_GUARDS_PASS" and
          proof.get("qualified_packer_sha256") == owner["packer_sha256"] and
          proof.get("reference_sha256") == "839a8782b73749da073e563863dfa9fa993d7384a874c8b912eb1db560e65c34" and
          proof.get("cases", 0) >= 4 and proof.get("finished") and
          all(row.get("exit_code") == 0 for row in proof.get("commands", [])),
          "Actual frozen output16 byte proof is required before the onebank load")
    # The full payload hash remains a receipt-bound no-rehash admission. The
    # actual OneBankStore repeats all digest/stat checks before allocating.
    integrity = _read_json(Path(owner["integrity_receipt_path"]), owner["integrity_receipt_sha256"])
    _need(integrity.get("phase") == "FULL_PAYLOAD_SHA256_VERIFIED_UNCHANGED_STATS" and
          integrity.get("bundle_root") == str(bundle_root) and
          integrity.get("manifest_sha256") == owner["manifest_sha256"] and
          integrity.get("no_weight_changes") is True,
          "Pinned full-payload integrity receipt does not match this bundle")


def _safe_output_root(path: Path) -> Path:
    root = path.absolute()
    _need(root.parent.resolve() == OUTPUT_BASE.resolve() or root.resolve().is_relative_to(OUTPUT_BASE.resolve()),
          "Run output must stay under the canonical pair-controller outputs directory")
    _need(not root.exists() and not root.is_symlink(), "Run output directory must be fresh")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    _need(not parent.is_symlink() and parent.is_dir(), "Output parent must be a real directory")
    root.mkdir(mode=0o700)
    return root


def _lock_foreground():
    q = WORK / "deepseek41-1017-nvfp4"
    d = WORK / "deepseek41"
    paths = [
        q / "state" / "owner.lock",
        MULTI / "state" / "coordinator.lock",
        d / "state" / "selected-speed-owner-v289.lock",
        d / "state" / "selected-server-v270.lock",
        WORKSPACE_ROOT / "Swift Flash Next MultiGPU" / "work" / "swift-flash-next-multigpu" / "state" / "coordinator.lock",
        WORKSPACE_ROOT / "Qwen3 Coder Next" / "work" / "qwen3-coder-next-speed" / "state" / "GPU0-probe.lock",
        WORKSPACE_ROOT / "RVN Qwen3.8 Flash Next" / "outputs" / "qwenflash-next-speed" / "launchers-speed-v10" / "speed-v10-console.lock",
        WORKSPACE_ROOT / "Shared" / "work" / "strata-server" / "state" / "devices" / (GPU0 + ".lock"),
        WORKSPACE_ROOT / "Shared" / "work" / "strata-server" / "state" / "devices" / (GPU1 + ".lock"),
        MULTI / "state" / "pair-controller-v1.lock",
    ]
    fds = []
    try:
        for path in paths:
            _need(path.parent.is_dir() and not path.parent.is_symlink(),
                  f"Required canonical lease directory is missing or redirected: {path.parent}")
            _need(not path.is_symlink(), f"Refusing shared lease symlink: {path}")
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise ControllerError(f"Canonical project/GPU lease is busy: {path}") from exc
            fds.append(fd)
        return fds
    except BaseException:
        for fd in reversed(fds):
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _assert_single_thread_before_fork() -> None:
    tasks = [entry.name for entry in Path("/proc/self/task").iterdir() if entry.name.isdecimal()]
    _need(len(tasks) == 1,
          f"The CUDA-empty onebank parent must be single-threaded before fork; found {len(tasks)} threads")


def _body_paths_and_hashes(req: dict):
    bodies = {}
    for job in req["jobs"]:
        p = Path(job["body_path"])
        _need(p.is_absolute() and p.resolve().is_relative_to(PROJECT.resolve()),
              f"Request body must be inside the canonical workspace: {p}")
        raw = _read_pinned_bytes(p, job["body_sha256"], limit=MAX_JSON_BYTES)
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerError(f"Invalid request body JSON: {p}") from exc
        _validate_body(body, job)
        bodies[job["id"]] = body
    return bodies


def _verify_pair_build_and_packet(req: dict):
    receipt, libraries, prefill_libraries, component_proof = _validate_build_receipt(req)
    expected = req["runtime"].get("expected_adapter_seal_sha256")
    seal_sha = _verify_pair_adapter(Path(req["runtime"]["pair_adapter_root"]), 0, expected)
    return receipt, libraries, prefill_libraries, component_proof, seal_sha


def _run_request_jobs(req: dict, bodies: dict, store, libraries: dict, prefill_libraries: dict,
                      adapter_root: Path, adapter_sha: str, outroot: Path, gpu_memory_samples: list,
                      cancel_event, request_sha256: str, owner_report: dict, maintenance=None):
    """Run one explicit request on an already-loaded bank; retain the bank on failure."""
    run_state = outroot / "state.json"
    gpu_memory_path = outroot / "GPU-memory.json"
    results = []
    summaries_by_fixture = {}
    child = None
    current_job = None
    last_reaped_v174=None
    try:
        if maintenance is not None:
            _need(all(job["pair_prefill"] == "0" for job in req["jobs"]), "Maintenance mode refuses unqualified pair-prefill DSO")
        _atomic_json(outroot / "CPU-owner.json", owner_report)
        for job in req["jobs"]:
            if cancel_event.is_set():
                raise ControllerError("Cancelled before next job")
            current_job = job
            context = job["context"]
            if job["pair_prefill"] == "1":
                _need(context in prefill_libraries, f"Missing admitted pair-prefill DSO for {context}")
                library_path, library_sha = prefill_libraries[context]
            else:
                library_path, library_sha = libraries[context]
            job_out = outroot / "jobs" / job["id"]
            job_out.parent.mkdir(parents=True, exist_ok=True)
            _need(not job_out.exists() and not job_out.is_symlink(), f"Job output must be fresh: {job['id']}")
            job_out.mkdir(mode=0o700)
            bank_binding=maintenance.assert_run(req,adapter_root,adapter_sha,libraries)if maintenance is not None else None
            store.validate_selected_owner(deep=True)
            _require_pair_compute_idle()
            _record_pair_memory(gpu_memory_path, gpu_memory_samples, f"before-child-{job['id']}")
            _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "GPU_CHILD_RUNNING",
                                     "run_id": req["run_id"], "request_sha256": request_sha256,
                                     "pair_adapter_seal_sha256": adapter_sha,
                                     "pair_build_receipt_sha256": req["runtime"]["pair_build_receipt_sha256"],
                                     "pid": os.getpid(), "owner_pid": os.getpid(), "GPU_child": None,
                                     "current_job": job["id"], "current_context": context,
                                     "cache_mode": job["cache_mode"], "cache_gib": job["cache_gib"],
                                     "mtp": job["mtp"], "pair_prefill": job["pair_prefill"],
                                     "CPU_group_sum": _cpu_group_sum(job),
                                     "one_bank_live": True, "jobs_total": len(req["jobs"]),
                                     "jobs_complete": len(results), "quality_admitted": False})
            _assert_single_thread_before_fork()
            child = os.fork()
            if child == 0:
                os.chdir(MULTI)
                code = _child_run(job, bodies[job["id"]], store, library_path, library_sha, adapter_root,
                                  cancel_event, job_out,bank_binding)
                os._exit(code)
            _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "GPU_CHILD_RUNNING",
                                     "run_id": req["run_id"], "request_sha256": request_sha256,
                                     "pair_adapter_seal_sha256": adapter_sha, "pid": os.getpid(),
                                     "owner_pid": os.getpid(), "GPU_child": child, "current_job": job["id"],
                                     "one_bank_live": True, "jobs_total": len(req["jobs"]),
                                     "jobs_complete": len(results), "quality_admitted": False})
            status = _wait_child(child, cancel_event, grace_seconds=30, state_path=run_state, job_id=job["id"],
                                 memory_callback=lambda label: _record_pair_memory(gpu_memory_path, gpu_memory_samples, label))
            reaped_pid_v174=child;child=None # waitpid already consumed it; never signal/reap a reused PID after receipt IO failure
            reap_record=dict(schema=174,phase="CHILD_REAPED",parent_pid=os.getpid(),child_pid=reaped_pid_v174,job_id=job["id"],job_output=str(job_out),runtime_dso_path=str(library_path),runtime_dso_sha256=library_sha,bank_format_binding=bank_binding,normal_exit=os.WIFEXITED(status),exit_code=os.WEXITSTATUS(status) if os.WIFEXITED(status) else None,signal=os.WTERMSIG(status) if os.WIFSIGNALED(status) else None,raw_wait_status=status,result_accepted=False)
            _atomic_json(job_out/"reap-proof.json",reap_record)
            close_path=job_out/"close-proof.json"
            last_reaped_v174=dict(record=reap_record,reap_proof=dict(path=str(job_out/"reap-proof.json"),sha256=_sha_file(job_out/"reap-proof.json",max_bytes=MAX_RECEIPT_BYTES)),close_proof_sha256=_sha_file(close_path,max_bytes=MAX_RECEIPT_BYTES) if close_path.is_file() else None)
            child = None
            _need(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0,
                  f"Pair GPU child failed for {job['id']}; result evidence remains in {job_out}")
            _require_pair_compute_idle()
            _record_pair_memory(gpu_memory_path, gpu_memory_samples, f"after-child-{job['id']}")
            _need((job_out/"close-proof.json").is_file(),"Successful child lacks durable close proof")
            result_path = job_out / "result.json"
            summary_path = job_out / "summary.json"
            _need(result_path.is_file() and summary_path.is_file() and not (job_out / "close-failure.json").exists(),
                  f"Child result or close proof missing for {job['id']}")
            summary = _read_json(summary_path, _sha_file(summary_path, max_bytes=MAX_JSON_BYTES), limit=MAX_JSON_BYTES)
            summary.update({"runtime_dso_path": str(library_path), "runtime_dso_sha256": library_sha})
            results.append(summary)
            fixture_key = (context, job["fixture_id"])
            arm = _job_arm(job)
            summaries_by_fixture.setdefault(fixture_key, {})[arm] = summary
            store.validate_selected_owner(deep=True)
            _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "JOB_COMPLETE",
                                     "run_id": req["run_id"], "request_sha256": request_sha256,
                                     "pair_adapter_seal_sha256": adapter_sha, "pid": os.getpid(),
                                     "owner_pid": os.getpid(), "GPU_child": None, "last_job": job["id"],
                                     "one_bank_live": True, "jobs_total": len(req["jobs"]),
                                     "jobs_complete": len(results), "quality_admitted": False})
            print(json.dumps({"phase": "JOB_COMPLETE", "job_id": job["id"], "context": context,
                              "cache_mode": job["cache_mode"], "cache_gib": job["cache_gib"],
                              "mtp": job["mtp"], "pair_prefill": job["pair_prefill"],
                              "CPU_group_sum": _cpu_group_sum(job),
                              "generation_tps": summary["generation_tps"], "new_prompt_tps": summary["new_prompt_tps"]},
                             ensure_ascii=False), flush=True)

        for _, arms in summaries_by_fixture.items():
            for arm, summary in arms.items():
                mode, gib, pair_prefill, mtp, cpu_group_sum = arm
                baseline = arms.get(("off", 0, pair_prefill, mtp, cpu_group_sum))
                if baseline is not None and arm != ("off", 0, pair_prefill, mtp, cpu_group_sum):
                    summary["cache0_matched_output_ids_sha256"] = baseline["output_token_ids_sha256"]
                    summary["exact_output_ids_match_cache0_same_mtp_prefill"] = summary["output_token_ids_sha256"] == baseline["output_token_ids_sha256"]
                if mtp == "native4":
                    mtp_off = arms.get((mode, gib, pair_prefill, "off", cpu_group_sum))
                    if mtp_off is not None:
                        summary["mtp_off_matched_output_ids_sha256"] = mtp_off["output_token_ids_sha256"]
                        summary["exact_output_ids_match_mtp_off_same_cache_prefill"] = summary["output_token_ids_sha256"] == mtp_off["output_token_ids_sha256"]
                if pair_prefill == "1":
                    prefill_off = arms.get((mode, gib, "0", mtp, cpu_group_sum))
                    if prefill_off is not None:
                        summary["pair_prefill_off_matched_output_ids_sha256"] = prefill_off["output_token_ids_sha256"]
                        summary["exact_output_ids_match_pair_prefill_off_same_cache_mtp"] = summary["output_token_ids_sha256"] == prefill_off["output_token_ids_sha256"]
                if cpu_group_sum == "1":
                    cpu_off = arms.get((mode, gib, pair_prefill, mtp, "0"))
                    if cpu_off is not None:
                        summary["CPU_group_sum_off_matched_output_ids_sha256"] = cpu_off["output_token_ids_sha256"]
                        summary["exact_output_ids_match_CPU_group_sum_off_same_cache_mtp_prefill"] = summary["output_token_ids_sha256"] == cpu_off["output_token_ids_sha256"]

        _atomic_json(outroot / "summary.json", {"schema": "deepseek1017-pair-controller-summary/1",
                                                 "phase": "SOURCE_RUNTIME_RESULTS_CAPTURED_QUALITY_UNACCEPTED",
                                                 "run_id": req["run_id"], "request_sha256": request_sha256,
                                                 "pair_adapter_seal_sha256": adapter_sha,
                                                 "pair_build_receipt_sha256": req["runtime"]["pair_build_receipt_sha256"],
                                                 "pair_prefill_build_receipt_sha256": req["runtime"].get("pair_prefill_build_receipt_sha256") if prefill_libraries else None,
                                                 "pair_prefill_component_receipt_sha256": req["runtime"].get("pair_prefill_component_receipt_sha256") if prefill_libraries else None,
                                                 "one_bank_bytes": owner_report["resident_bytes"],
                                                 "source_revision": owner_report["source_revision"],
                                                 "GPU_memory_samples": gpu_memory_samples,
                                                 "results": results, "quality_admitted": False,
                                                 "runtime_performance_admitted": False,
                                                 "cache_arithmetic_quality": "EXPERIMENTAL_UNACCEPTED_GPU_EXPERT_GEMV"})
        _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "REQUEST_COMPLETE_BANK_RETAINED",
                                 "run_id": req["run_id"], "request_sha256": request_sha256,
                                 "pid": os.getpid(), "owner_pid": os.getpid(), "GPU_child": None,
                                 "one_bank_live": True, "jobs_total": len(req["jobs"]),
                                 "jobs_complete": len(results), "quality_admitted": False})
        return results, None
    except BaseException as exc:
        if child is not None:
            cancel_event.set()
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
            _wait_child(child, cancel_event, grace_seconds=30, state_path=run_state,
                        job_id=current_job["id"] if current_job else "unknown")
            child = None
        failure = {"error": str(exc), "traceback": traceback.format_exc(),
                   "failed_job": current_job["id"] if current_job else None,
                   "reaped_child":last_reaped_v174,
                   "bank_retained": True}
        _atomic_json(outroot / "failure.json", failure)
        _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "REQUEST_FAILED_BANK_RETAINED",
                                 "run_id": req.get("run_id"), "request_sha256": request_sha256,
                                 "pid": os.getpid(), "owner_pid": os.getpid(), "GPU_child": None,
                                 "current_job": current_job.get("id") if current_job else None,
                                 "one_bank_live": True, "jobs_total": len(req.get("jobs", [])),
                                 "jobs_complete": len(results), **failure, "quality_admitted": False})
        print(json.dumps({"phase": "REQUEST_FAILED_BANK_RETAINED", "error": str(exc),
                          "failed_job": failure["failed_job"]}, ensure_ascii=False), file=sys.stderr, flush=True)
        return results, failure


def run_controller(args) -> int:
    _need(args.execute, "Root must explicitly pass --execute; this controller has no daemon mode")
    _need(os.environ.get("DS1017_PAIR_CONTROLLER_ROOT_EXECUTION") == "1",
          "Root must set DS1017_PAIR_CONTROLLER_ROOT_EXECUTION=1 after taking the GPU0/GPU1 lease")
    maintenance_path=getattr(args,"maintenance_config",None);maintenance_sha=getattr(args,"maintenance_config_sha256",None);preload_only=getattr(args,"preload_only",False)
    _need((maintenance_path is None)==(maintenance_sha is None),"Explicit maintenance config path+SHA required together")
    _need(not getattr(args,"allow_profile_admission_v163",False) or maintenance_path is not None,"Profile admission opt-in requires maintenance-aware fresh parent")
    _need(not getattr(args,"allow_revalidate_after_reaped_child_v174",False) or maintenance_path is not None,"Child revalidation opt-in requires maintenance-aware fresh parent")
    _need(not maintenance_path or preload_only,"Maintenance-aware fresh parent requires explicit --preload-only before its first child")
    _need(not preload_only or(maintenance_path is not None and args.hold_open),"Preload-only requires maintenance config and held-open private stdin")
    _check_parent_environment()
    _require_pair_compute_idle()
    req_path = Path(args.request).absolute()
    _need(req_path.resolve().is_relative_to(PROJECT.resolve()), "Request JSON must be inside the canonical DeepSeek model section")
    req = _load_request(req_path, args.request_sha256)
    if maintenance_path is not None:
        _need(all(job["pair_prefill"] == "0" for job in req["jobs"]), "Maintenance mode refuses unqualified pair-prefill DSO")
    bodies = _body_paths_and_hashes(req)
    _preflight_owner_receipts(req["owner"])
    build_receipt, libraries, prefill_libraries, prefill_component_proof = _validate_build_receipt(req)
    outroot = _safe_output_root(Path(args.output_dir))
    gpu_memory_path = outroot / "GPU-memory.json"
    gpu_memory_samples = []
    run_state = outroot / "state.json"
    shared_cancel = __import__("multiprocessing").get_context("fork").Event()
    cancel_requested = False
    class CancelState:
        def is_set(self):
            return cancel_requested or shared_cancel.is_set()
        def set(self):
            # Called only from normal control flow, never from a signal handler.
            shared_cancel.set()
    cancel_event = CancelState()
    old_handlers = {}
    def request_cancel(_signum, _frame):
        nonlocal cancel_requested
        # Signal handlers must not acquire the multiprocessing.Event condition lock.
        cancel_requested = True
    lease_fds = _lock_foreground()
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_cancel)
    store = None
    maintenance = None
    native_children_clean = [True]
    child = None
    current_job = None
    results = []
    failure = None
    bank_release_error = None
    started = time.time()
    try:
        # Shared project/GPU leases are held before this second idle check and
        # before any cold CPU-bank allocation.
        _require_pair_compute_idle()
        _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "PREFLIGHT_PASS",
                                 "run_id": req["run_id"], "request_sha256": args.request_sha256,
                                 "pair_build_receipt_sha256": req["runtime"]["pair_build_receipt_sha256"],
                                 "pid": os.getpid(), "GPU_child": None, "one_bank_live": False,
                                 "jobs_total": len(req["jobs"]), "jobs_complete": 0,
                                 "quality_admitted": False})
        _record_pair_memory(gpu_memory_path, gpu_memory_samples, "before-cpu-bank-preload")
        bank = req["owner"]
        owner_mod = _import_owner_module()
        store = owner_mod.OneBankStore(Path(bank["bundle_root"]), bank["manifest_sha256"])
        _need(store.manifest.get("source_revision") == bank["source_revision"], "New1017 source revision differs")
        _need(not store.fixture, "Production pair controller refuses fixture/fake stores")
        _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "CPU_BANK_PRELOAD",
                                 "run_id": req["run_id"], "request_sha256": args.request_sha256,
                                 "pid": os.getpid(), "owner_pid": os.getpid(), "GPU_child": None,
                                 "one_bank_live": False, "jobs_total": len(req["jobs"]), "jobs_complete": 0,
                                 "started_unix": started, "quality_admitted": False})
        last_preload_sample = [time.monotonic()]
        def on_preload_progress(item):
            print(json.dumps(item, ensure_ascii=False), flush=True)
            if time.monotonic() - last_preload_sample[0] >= 30.0:
                _record_pair_memory(gpu_memory_path, gpu_memory_samples, "cpu-bank-preload-periodic")
                last_preload_sample[0] = time.monotonic()
        report = store.preload_selected(
            Path(bank["packer_path"]), bank["packer_sha256"],
            reserve_bytes=80 * GIB, budget_bytes=432 * GIB,
            integrity_receipt=Path(bank["integrity_receipt_path"]),
            integrity_sha256=bank["integrity_receipt_sha256"], workers=8,
            cancel=cancel_event,
            on_progress=on_preload_progress,
        )
        store.validate_selected_owner(deep=True)
        _need(report.get("runtime_admissible") is True and report.get("resident_bytes") == 435110212200,
              "The exact full 435,110,212,200-byte OneBankStore was not prepared")
        _need(report.get("all_pack_workers_joined") is True and report.get("pack_workers") == 8,
              "All eight CPU packers must join before the first fork")
        _need("torch" not in sys.modules and not any("libcuda.so" in row for row in Path("/proc/self/maps").read_text().splitlines()),
              "CPU bank parent acquired an accelerator runtime")
        _atomic_json(outroot / "CPU-owner.json", report)
        _record_pair_memory(gpu_memory_path, gpu_memory_samples, "after-cpu-bank-preload")
        if maintenance_path is not None:
            _assert_single_thread_before_fork()
            def revalidation_outer_exclusion_v174():
                _need(child is None,"Outer child not reaped")
                for fd in lease_fds:fcntl.fcntl(fd,fcntl.F_GETFD)
            def maintenance_exclusion():
                _need(native_children_clean[0],"Prior GPU child lacks complete close/drain proof; maintenance owner remains inadmissible")
                revalidation_outer_exclusion_v174()
            maintenance=Manager(store,owner_mod,maintenance_path,maintenance_sha,outroot/"maintenance",_atomic_json,
                                _require_pair_idle_recovery_v146,maintenance_exclusion,
                                emit=lambda line:print(line,flush=True),
                                allow_profile_admission_v163=getattr(args,"allow_profile_admission_v163",False),
                                profile_validator_v163=_validate_admission_profile_v163)

        # Permit the expensive CPU-only owner preload to finish before the pair
        # runtime source packet is frozen, but do not launch a GPU child early.
        runtime_cfg = req["runtime"]
        adapter_root = Path(runtime_cfg["pair_adapter_root"]).absolute()
        adapter_sha = _verify_pair_adapter(adapter_root, runtime_cfg["adapter_wait_seconds"],
                                           runtime_cfg["expected_adapter_seal_sha256"],
                                           cancel_event=cancel_event)
        _need(not cancel_event.is_set(), "Cancelled before the first pair child")

        current_req, current_request_sha, current_bodies = req, args.request_sha256, bodies
        current_build_receipt, current_libraries = build_receipt, libraries
        current_prefill_libraries = prefill_libraries
        current_outroot, current_memory_path, current_memory_samples = outroot, gpu_memory_path, gpu_memory_samples
        initial_owner_binding = {key: req["owner"][key] for key in (
            "bundle_root", "manifest_sha256", "source_revision", "packer_path", "packer_sha256",
            "packer_proof_path", "packer_proof_sha256", "integrity_receipt_path", "integrity_receipt_sha256")}
        request_count = 0
        had_request_failure = False
        skip_initial=preload_only
        closing=False
        while True:
            if not skip_initial:
                run_results, request_failure = _run_request_jobs(
                    current_req, current_bodies, store, current_libraries, current_prefill_libraries,
                    adapter_root, adapter_sha, current_outroot, current_memory_samples, cancel_event,
                    current_request_sha, report,maintenance)
                results.extend(run_results);request_count+=1
                if request_failure is not None:
                    failure=request_failure;had_request_failure=True;native_children_clean[0]=False
                    if maintenance is not None:maintenance.block_after_child_failure()
            skip_initial=False
            if not args.hold_open or cancel_event.is_set():break
            # Stay in this private command loop after status/rejection/maintenance.
            # Only a fully admitted new run leaves it; never rerun prior request.
            while not cancel_event.is_set():
                state=dict(schema="deepseek1017-pair-controller-state/146",phase="OWNER_READY",run_id=current_req["run_id"],
                           request_sha256=current_request_sha,pid=os.getpid(),owner_pid=os.getpid(),GPU_child=None,one_bank_live=True,
                           requests_processed=request_count,jobs_complete_total=len(results),last_request_failure=failure,quality_admitted=False)
                if maintenance is not None:
                    state['maintenance']=maintenance.status()
                    if not state['maintenance']['runtime_admissible']:state['phase']='OWNER_MAINTENANCE_BLOCKED'
                _atomic_json(current_outroot/"state.json",state)
                print(json.dumps(dict(state,one_bank_bytes=report['resident_bytes'],next_input='private JSON status/close/run/maintenance_relayout/admit_format_profile'),ensure_ascii=False),flush=True)
                while not cancel_event.is_set():
                    readable,_,_=select.select([sys.stdin],[],[],1.0)
                    if readable:break
                if cancel_event.is_set():break
                line=sys.stdin.readline()
                if not line:closing=True;break
                try:
                    _need(len(line.encode('utf-8'))<=16384,'command exceeds size bound')
                    command=json.loads(line);_need(isinstance(command,dict),'command must be object');action=command.get('action')
                    if action=='close':
                        _need(set(command)=={'action'},'exact close command required');closing=True;break
                    if action=='status':
                        _need(set(command)=={'action'},'exact status command required');continue
                    if action=='revalidate_after_reaped_child':
                        _need(maintenance is not None,'failed-child revalidation requires maintenance-aware parent')
                        from revalidate_v174 import revalidate_after_reaped_child
                        receipt=revalidate_after_reaped_child(maintenance,command,failure,native_children_clean,allowed=getattr(args,'allow_revalidate_after_reaped_child_v174',False),outer_check=revalidation_outer_exclusion_v174)
                        print(json.dumps(receipt,sort_keys=True),flush=True);continue
                    if action=='admit_format_profile':
                        _need(maintenance is not None,'profile admission requires maintenance-aware parent')
                        print(json.dumps(maintenance.admit_profile_v163(command),sort_keys=True),flush=True);continue
                    if action=='maintenance_relayout':
                        _need(maintenance is not None,'maintenance disabled in ordinary controller mode')
                        maintenance.maintain(command,cancel_event);report=dict(report,bank_format_binding_v146=maintenance.binding());continue
                    _need(action=='run'and set(command)=={'action','request_path','request_sha256','output_dir'},'expected exact private status/close/run/maintenance command')
                    next_request_sha=command['request_sha256'];_need(isinstance(next_request_sha,str)and bool(SHA_RE.fullmatch(next_request_sha)),'interactive request SHA invalid')
                    next_req_path=Path(command['request_path']).absolute();_need(next_req_path.resolve().is_relative_to(PROJECT.resolve()),'request must stay inside model section')
                    next_req=_load_request(next_req_path,next_request_sha)
                    if maintenance is not None:
                        _need(all(job["pair_prefill"] == "0" for job in next_req["jobs"]), "Maintenance mode refuses unqualified pair-prefill DSO")
                    next_owner={key:next_req['owner'][key]for key in initial_owner_binding}
                    _need(next_owner==initial_owner_binding,'interactive request changes loaded original owner/source/packer/integrity')
                    next_bodies=_body_paths_and_hashes(next_req);_preflight_owner_receipts(next_req['owner'])
                    next_build_receipt,next_libraries,next_prefill_libraries,_=_validate_build_receipt(next_req)
                    next_adapter_root=Path(next_req['runtime']['pair_adapter_root']).absolute()
                    next_adapter_sha=_verify_pair_adapter(next_adapter_root,0,next_req['runtime'].get('expected_adapter_seal_sha256'))
                    if maintenance is not None:maintenance.assert_run(next_req,next_adapter_root,next_adapter_sha,next_libraries)
                    next_outroot=_safe_output_root(Path(command['output_dir']));next_memory_path=next_outroot/'GPU-memory.json';next_memory_samples=[]
                    _require_pair_compute_idle();_record_pair_memory(next_memory_path,next_memory_samples,'reused-bank-before-request')
                    current_req,current_request_sha,current_bodies=next_req,next_request_sha,next_bodies
                    current_build_receipt,current_libraries=next_build_receipt,next_libraries;current_prefill_libraries=next_prefill_libraries
                    adapter_root,adapter_sha=next_adapter_root,next_adapter_sha
                    current_outroot,current_memory_path,current_memory_samples=next_outroot,next_memory_path,next_memory_samples
                    break
                except BaseException as command_error:
                    print(json.dumps(dict(phase='OWNER_COMMAND_REJECTED',error=str(command_error),maintenance=maintenance.status()if maintenance else None),ensure_ascii=False),flush=True)
            if closing or cancel_event.is_set():break
        return 1 if had_request_failure else 0
    except BaseException as exc:
        failure = {"error": str(exc), "traceback": traceback.format_exc()}
        if child is not None:
            cancel_event.set()
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
            status = _wait_child(child, cancel_event, grace_seconds=30, state_path=run_state, job_id=current_job["id"] if current_job else "unknown")
            child = None
        _atomic_json(run_state, {"schema": "deepseek1017-pair-controller-state/1", "phase": "FAILED_OR_CANCELLED",
                                 "run_id": req.get("run_id"), "request_sha256": args.request_sha256,
                                 "pid": os.getpid(), "owner_pid": os.getpid(), "GPU_child": None,
                                 "current_job": current_job.get("id") if current_job else None,
                                 "one_bank_live": store is not None and not getattr(store, "_closed", True),
                                 "jobs_total": len(req.get("jobs", [])), "jobs_complete": len(results),
                                 **failure, "quality_admitted": False})
        print(json.dumps({"phase": "FAILED_OR_CANCELLED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            if child is not None:
                cancel_event.set()
                try:
                    os.kill(child, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                _wait_child(child, cancel_event, grace_seconds=30, state_path=run_state,
                            job_id=current_job["id"] if current_job else "unknown")
                child = None
            if store is not None:
                try:
                    # Every fresh child has been reaped before the sole parent
                    # releases the shared bank; a foreign GPU PID already
                    # stopped the queue before this cleanup path.
                    if maintenance is not None:maintenance.prepare_close()
                    store.close()
                    if maintenance is not None:maintenance.close_after_bank()
                except BaseException as exc:
                    bank_release_error = {"error": str(exc), "traceback": traceback.format_exc()}
            terminal = ("FAILED_BANK_NOT_RELEASED" if bank_release_error is not None else
                        "OWNER_CANCELLED_BANK_RELEASED" if cancel_event.is_set() else
                        "OWNER_CLOSED_AFTER_REQUEST_FAILURES" if failure is not None else
                        "OWNER_CLOSED_BANK_RELEASED")
            final_state_path = (current_outroot / "state.json") if "current_outroot" in locals() else run_state
            _atomic_json(final_state_path, {"schema": "deepseek1017-pair-controller-state/1", "phase": terminal,
                                     "run_id": current_req.get("run_id") if "current_req" in locals() else req.get("run_id"),
                                     "request_sha256": current_request_sha if "current_request_sha" in locals() else args.request_sha256,
                                     "pair_adapter_seal_sha256": adapter_sha if 'adapter_sha' in locals() else None,
                                     "pid": os.getpid(), "owner_pid": os.getpid(), "GPU_child": None,
                                     "one_bank_live": bank_release_error is not None,
                                     "requests_processed": request_count if "request_count" in locals() else 0,
                                     "jobs_complete_total": len(results),
                                     "finished_unix": time.time(), "failure": failure,
                                     "bank_release_failure": bank_release_error,
                                     "quality_admitted": False})
        finally:
            for fd in reversed(lease_fds):
                os.close(fd)
            for sig, old in old_handlers.items():
                signal.signal(sig, old)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Root foreground controller; one CPU bank, one fresh paired-GPU child at a time")
    parser.add_argument("--execute", action="store_true", help="explicit Root opt-in; never starts a daemon")
    parser.add_argument("--request", required=True, type=Path, help="explicit pinned workload JSON; no globbing")
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path, help="fresh directory under canonical pair-controller outputs")
    parser.add_argument("--close-after-request", dest="hold_open", action="store_false", default=True,
                        help="release the CPU bank after the initial request instead of staying ready for stdin commands")
    parser.add_argument("--maintenance-config",type=Path,help="fresh-parent Root-pinned explicit maintenance proof configuration; default disabled")
    parser.add_argument("--maintenance-config-sha256")
    parser.add_argument("--preload-only",action="store_true",help="maintenance-capable fresh parent loads old16 once and awaits explicit format-bound workload")
    parser.add_argument("--allow-profile-admission-v163",action="store_true",help="explicit Root opt-in for pinned append-only format profiles in this fresh maintenance parent")
    parser.add_argument("--allow-revalidate-after-reaped-child-v174",action="store_true",help="Root explicit default-OFF failed-child bank revalidation; no automatic replay or GPU reset")
    args = parser.parse_args(argv)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
