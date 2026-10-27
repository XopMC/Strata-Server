"""Our bounded group quantizer with diagonal activation importance.

This is an explicitly versioned candidate representation, not native MXFP4 or
GGML Q4_0. It does not establish model quality. Source FP8/MXFP4 must first be
decoded with their original scales, and every candidate needs held-out checks.
"""
from dataclasses import dataclass
import numpy as np


FORMAT = 'deepseek41-signed-uniform-bf16-group-v1'


def bf16_scales(values):
    values = np.asarray(values,dtype='<f4')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Quantization scales must be finite and nonnegative')
    words = values.view('<u4')
    rounded = ((words+0x7fff+((words>>16)&1))>>16).astype('<u2')
    decoded = (rounded.astype('<u4')<<16).view('<f4')
    if not np.isfinite(decoded).all():
        raise ValueError('BF16 scale overflow')
    return rounded,decoded


def pack_codes(codes,bits):
    values = np.asarray(codes)
    if bits not in (2,3,4,6,8) or values.ndim != 2 or values.dtype.kind not in 'iu':
        raise ValueError('Invalid group quantization codes/bits')
    if np.any(values >= (1<<bits)-1) or np.any(values < 0):
        raise ValueError('Reserved/out-of-range signed-uniform code')
    rows,width = values.shape
    groups = (width+7)//8
    padded = np.zeros((rows,groups*8),dtype=np.uint64)
    padded[:,:width] = values
    shifts = np.arange(8,dtype=np.uint64)*bits
    words = (padded.reshape(rows,groups,8)<<shifts).sum(-1,dtype=np.uint64).astype('<u8')
    # Eight codes occupy exactly `bits` bytes. Copy before flattening because
    # the upper unused bytes of each uint64 word are interleaved otherwise.
    return words.view(np.uint8).reshape(rows,groups,8)[:,:,:bits].copy().reshape(rows,-1)[:,:((width*bits+7)//8)].copy()


def unpack_codes(packed,bits,width):
    packed = np.asarray(packed)
    if bits not in (2,3,4,6,8) or type(width) is not int or width < 1 or packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError('Invalid packed candidate dimensions')
    if packed.shape[1] != (width*bits+7)//8:
        raise ValueError('Packed candidate row byte size disagrees')
    positions = np.arange(width,dtype=np.int64)*bits
    codes = np.zeros((len(packed),width),dtype=np.uint8)
    for shift in range(bits):
        codes |= ((packed[:,(positions+shift)//8]>>((positions+shift)%8))&1).astype(np.uint8)<<shift
    if np.any(codes == (1<<bits)-1):
        raise ValueError('Reserved signed-uniform code in candidate')
    return codes


@dataclass(frozen=True)
class QuantizedRows:
    packed: np.ndarray
    scales: np.ndarray
    bits: int
    width: int
    group_size: int
    weighted_squared_error: float
    weighted_source_energy: float

    @property
    def bytes(self):
        return self.packed.nbytes+self.scales.nbytes

    def decode(self):
        codes = unpack_codes(self.packed,self.bits,self.width).astype(np.int16)-((1<<(self.bits-1))-1)
        scale = (self.scales.astype('<u4')<<16).view('<f4')
        return (codes.reshape(len(codes),-1,self.group_size).astype(np.float32)*scale[...,None]).reshape(len(codes),self.width)


def quantize_rows(weights,bits,importance=None,*,group_size=32,scratch_budget_bytes=64*1024**2):
    """Fit each scale to weighted distortion; keep the best actual BF16 fit.

    Clipping candidates and three fixed-code least-squares refinements are
    evaluated against the same source values and diagonal calibration moments.
    BF16-rounded scales enter the objective; an unrepresentable ideal scale
    never receives credit. Unobserved importance must be declared as None.
    """
    weights = np.asarray(weights)
    if (weights.ndim != 2 or weights.dtype != np.float32 or not weights.size or
            bits not in (2,3,4,6,8) or group_size != 32 or weights.shape[1]%group_size):
        raise ValueError('Candidate requires finite F32 rows with complete32-value groups')
    if weights.nbytes*24 > scratch_budget_bytes:
        raise ValueError('Quantizer tile exceeds its conservative scratch budget')
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite source quantization tile')
    if importance is None:
        importance = np.ones(weights.shape[1],dtype=np.float64)
    importance = np.asarray(importance,dtype=np.float64)
    if importance.shape != (weights.shape[1],) or not np.isfinite(importance).all() or np.any(importance < 0) or not np.any(importance > 0):
        raise ValueError('Invalid per-column activation importance')
    importance = importance.reshape(1,-1,group_size)
    source = weights.reshape(len(weights),-1,group_size).astype(np.float64)
    qmax = (1<<(bits-1))-1
    maximum = np.abs(source).max(-1)
    best_error = np.full(maximum.shape,np.inf)
    best_codes = np.zeros(source.shape,dtype=np.int16)
    best_scales = np.zeros(maximum.shape,dtype='<u2')
    weighted_rms = np.sqrt(np.divide((source**2*importance).sum(-1),importance.sum(-1),
        out=np.zeros_like(maximum),where=importance.sum(-1) > 0))
    starts = [maximum*clipping/qmax for clipping in (1.,.995,.98,.95,.9,.85,.8,.7)]
    # A low-importance outlier can leave all important codes zero when every
    # start uses the absolute maximum. Weighted RMS provides distinct basins.
    starts += [weighted_rms*factor/qmax for factor in (1.,2.,3.,4.)]
    for initial in starts:
        current = initial
        for iteration in range(4):
            packed_scale,scale = bf16_scales(current.astype(np.float32))
            denominator = np.where(scale > 0,scale,1.)[...,None]
            signed = np.clip(np.rint(source/denominator),-qmax,qmax).astype(np.int16)
            error = ((source-signed*scale[...,None])**2*importance).sum(-1)
            improved = error < best_error
            best_error = np.where(improved,error,best_error)
            best_codes[improved] = signed[improved]
            best_scales[improved] = packed_scale[improved]
            square = (signed.astype(np.float64)**2*importance).sum(-1)
            current = np.divide((source*signed*importance).sum(-1),square,
                out=np.zeros_like(square),where=square > 0)
    unsigned = (best_codes+qmax).astype(np.uint8).reshape(weights.shape)
    return QuantizedRows(pack_codes(unsigned,bits),best_scales,bits,weights.shape[1],group_size,
        float(best_error.sum()),float((source**2*importance).sum()))


# These are candidates, not quality-accepted or selected profiles. Critical
# attention/router/shared/HC/vision/MTP tensors retain source representation.
PROFILES = {
    'quality': {'engram_bits':6,'experts':'native-mxfp4','protected_lookup_coverage':.999},
    'balance': {'engram_bits':4,'experts':'native-mxfp4','protected_lookup_coverage':.99},
    'speed-memory': {'engram_bits':3,'experts':'sensitivity-selected','protected_lookup_coverage':.95,
        'expert_bits':3,'expert_maximum_fraction':.75,'expert_maximum_relative_error':.20},
    'extreme': {'engram_bits':2,'experts':'sensitivity-selected','protected_lookup_coverage':.90,
        'expert_bits':2,'expert_maximum_fraction':.75,'expert_maximum_relative_error':.40},
}
