"""CPU decoding of the checkpoint's original FP8/MXFP4 representations."""
import numpy as np


def _e4m3_formula(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    exponent = (codes >> 3) & 15
    mantissa = codes & 7
    values = np.where(exponent == 0, mantissa.astype(np.float32) * 2**-9,
                      np.ldexp(1 + mantissa.astype(np.float32) / 8,
                               exponent.astype(np.int16) - 7))
    values = np.where((exponent == 15) & (mantissa == 7), np.nan, values)
    return np.copysign(values, np.where(codes & 128, -1.0, 1.0)).astype(np.float32)


def _e8m0_formula(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    exponent = np.where(codes == 255, 127, codes).astype(np.int16) - 127
    values = np.ldexp(np.ones(codes.shape, dtype=np.float32), exponent)
    return np.where(codes == 255, np.nan, values).astype(np.float32)


# All256source codes are finite-domain lookups. Compute the exact existing
# float32 decoder results once rather than redoing array exponent/sign math
# for every row tile of the CPU baseline or the huge Engram quantizer.
_E4M3=_e4m3_formula(np.arange(256,dtype=np.uint8));_E4M3.setflags(write=False)
_E8M0=_e8m0_formula(np.arange(256,dtype=np.uint8));_E8M0.setflags(write=False)


def e4m3fn(codes):return _E4M3[np.asarray(codes,dtype=np.uint8)]
def e8m0(codes):return _E8M0[np.asarray(codes,dtype=np.uint8)]


def finite_scales(codes):
    scales = e8m0(codes)
    if not np.isfinite(scales).all():
        raise ValueError("Checkpoint contains a non-finite E8M0 scale")
    return scales


def mxfp4(packed, scales, block_size=32):
    packed = np.asarray(packed).view(np.uint8)
    if packed.ndim != 2 or packed.shape[1] * 2 % block_size:
        raise ValueError("MXFP4 matrix must have full block-aligned rows")
    expected = (packed.shape[0], packed.shape[1] * 2 // block_size)
    scales = np.asarray(scales, dtype=np.uint8)
    if scales.shape != expected:
        raise ValueError("MXFP4 scale shape does not match packed rows")
    positive = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    table = np.concatenate((positive, -positive))
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    values = table[codes].reshape(packed.shape[0], -1, block_size)
    return (values * finite_scales(scales)[:, :, None]).reshape(codes.shape)


def fp8_row_groups(codes, scales, block_size=32):
    codes = np.asarray(codes, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] % block_size:
        raise ValueError("FP8 rows must be block-aligned")
    if scales.shape != (codes.shape[0], codes.shape[1] // block_size):
        raise ValueError("FP8 row-group scale shape mismatch")
    return (e4m3fn(codes).reshape(codes.shape[0], -1, block_size)
            * finite_scales(scales)[:, :, None]).reshape(codes.shape)


def fp8_block_rows(codes, scales, source_rows, block_size=32):
    codes = np.asarray(codes, dtype=np.uint8)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    scales = np.asarray(scales, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[0] != len(source_rows) or scales.ndim != 2:
        raise ValueError("FP8 block input shape mismatch")
    rows = source_rows // block_size
    columns = np.arange(codes.shape[1]) // block_size
    if np.any(source_rows < 0) or np.any(rows >= scales.shape[0]) or np.any(columns >= scales.shape[1]):
        raise ValueError("FP8 block scales outside matrix bounds")
    selected = finite_scales(scales)[rows[:, None], columns[None, :]]
    return e4m3fn(codes) * selected


def bf16(raw):
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (words << 16).view(np.float32)
