"""Canonical FFI strides without copying logically contiguous cache prefixes."""
import torch


def canonical_tensor(tensor):
    if tensor.layout != torch.strided or tensor.numel() == 0:
        raise ValueError('Canonical layout requires a nonempty strided tensor')
    stride,extent = [],1
    for size in reversed(tensor.shape):
        stride.append(extent)
        extent *= size
    stride = tuple(reversed(stride))
    # A size1 dimension never changes an addressed element. Rewriting only
    # such strides preserves offsets/data and satisfies TVM's strict checks.
    if all(size == 1 or actual == expected for size,actual,expected in zip(tensor.shape,tensor.stride(),stride)):
        return tensor.as_strided(tensor.shape,stride)
    return tensor.contiguous()
