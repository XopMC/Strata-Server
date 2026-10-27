"""GGML Q4_0 compressed latent storage; index/window remain native BF16.

The scale and nibble arrays are split for CUDA access, but each group of32
contains precisely the reference FP16 scale and16packed bytes. No latent BF16
mirror is retained. Writes and speculative snapshots are bounded row ranges.
"""
import hashlib
from pathlib import Path
import torch

GROUP=32
BLOCK_BYTES=18


def quantize(values):
    if values.ndim<1 or values.shape[-1]%GROUP or values.dtype not in(torch.bfloat16,torch.float16,torch.float32):
        raise ValueError('Q4_0 requires floating rows with a32-aligned width')
    if not bool(torch.isfinite(values).all()):raise ValueError('Nonfinite Q4_0 cache input')
    blocks=values.float().unflatten(-1,(-1,GROUP))
    # GGML uses the SIGNED first largest absolute value, not positive absmax.
    positions=blocks.abs().argmax(-1,keepdim=True)
    scale=blocks.gather(-1,positions).squeeze(-1)/-8
    reciprocal=torch.where(scale!=0,scale.reciprocal(),0)
    codes=(blocks*reciprocal.unsqueeze(-1)+8.5).to(torch.int32).clamp(0,15).to(torch.uint8)
    stored_scale=scale.to(torch.float16)
    if not bool(torch.isfinite(stored_scale).all()):raise ValueError('Q4_0 FP16 scale overflow')
    packed=(codes[...,:16]|(codes[...,16:]<<4)).flatten(-2)
    return packed.contiguous(),stored_scale.contiguous()


def dequantize(codes,scales,*,dtype=torch.bfloat16):
    if codes.dtype!=torch.uint8 or scales.dtype!=torch.float16 or codes.shape[:-1]!=scales.shape[:-1]or codes.shape[-1]!=scales.shape[-1]*16:
        raise ValueError('Invalid split Q4_0 code/scale layout')
    packed=codes.unflatten(-1,(-1,16))
    values=torch.cat(((packed&15).to(torch.int16)-8,(packed>>4).to(torch.int16)-8),-1)
    return (values.float()*scales.float().unsqueeze(-1)).flatten(-2).to(dtype)


class Q4Cache:
    def __init__(self,capacity,width,device,*,batch=1):
        if type(capacity)is not int or capacity<1 or type(width)is not int or width%32 or width<32 or batch!=1:
            raise ValueError('Q4_0 cache is bounded single-batch aligned storage')
        self.capacity,self.width=capacity,width
        self.device=torch.device(device)
        self.codes=torch.full((1,capacity,width//2),136,dtype=torch.uint8,device=self.device)
        self.scales=torch.zeros((1,capacity,width//32),dtype=torch.float16,device=self.device)
        self.rows_written=0

    @property
    def storage_bytes(self):return self.codes.numel()+self.scales.numel()*2

    def write(self,first,values):
        if type(first)is not int or values.ndim!=3 or values.shape[0]!=1 or values.shape[-1]!=self.width or values.device!=self.device or first<0 or first+values.shape[1]>self.capacity:
            raise ValueError('Q4_0 cache write outside its bounded row range')
        codes,scales=quantize(values)
        self.codes[:,first:first+values.shape[1]].copy_(codes)
        self.scales[:,first:first+values.shape[1]].copy_(scales)
        self.rows_written+=values.shape[1]

    def raw_buffers(self,first,last):
        if not 0<=first<=last<=self.capacity:raise ValueError('Q4_0 raw snapshot outside capacity')
        return self.codes[:,first:last],self.scales[:,first:last]

    def prefix(self,length):
        if type(length)is not int or not 0<=length<=self.capacity:raise ValueError('Q4_0 prefix outside capacity')
        return Q4Prefix(self,length)


class Q4Prefix:
    """Descriptor only; sparse attention decodes requested64-position blocks."""
    def __init__(self,cache,length):self.cache,self.length=cache,length
    @property
    def shape(self):return(1,self.length,self.cache.width)
    @property
    def device(self):return self.cache.device
    @property
    def dtype(self):return torch.bfloat16


def cache_for(attention):return getattr(attention,'_deepseek41_q4_kv',None)


def implementation_sha256():
    digest=hashlib.sha256()
    for name in('q4_cache.py','q4_attention.py'):
        digest.update(name.encode());digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def admit_component(reference):
    root=Path(__file__).resolve().parents[3]
    import json
    proof=json.loads((root.parents[1]/'outputs/deepseek41/resident-lab/ram-isolated-q4-kv-v1.json').read_text())
    child=proof.get('child',{})
    if(child.get('implementation_sha256')!=implementation_sha256()or child.get('all_matches_dequantized_bf16')is not True or
       proof.get('memory_errors')!=0 or proof.get('source_revision')!=reference.store.ram_preload_report['source_revision']):
        raise ValueError('Q4_0 cache requires matching isolated pack/attention and memcheck proof')


def attention_forward(graph,attention,x,start):
    """Pinned Attention.forward with separate window/packed latent inputs."""
    from .q4_attention import sparse_attn
    batch,length,_=x.shape;frequencies=attention.freqs_cis[start:start+length]
    rd=attention.rope_head_dim
    qr=attention.q_norm(attention.wq_a(x))
    query=attention.wq_b(qr).unflatten(-1,(attention.n_local_heads,attention.head_dim))
    graph.apply_rotary_emb(query[...,-rd:],frequencies)
    window,indices=attention._window_kv(x,frequencies,start)
    packed,compressed_indices=attention._compress_kv(x,qr,start,window.shape[1])
    if not isinstance(packed,Q4Prefix):raise ValueError('Q4_0 attention lost its packed owner')
    indices=torch.cat((indices,compressed_indices),dim=-1)
    output=sparse_attn(query,window,packed,attention.attn_sink,indices,attention.softmax_scale)
    graph.apply_rotary_emb(output[...,-rd:],frequencies,True)
    output=output.view(batch,length,attention.n_local_groups,-1)
    weights=attention.wo_a.weight.view(attention.n_local_groups,attention.o_lora_rank,-1)
    output=torch.einsum('bsgd,grd->bsgr',output,weights)
    return attention.wo_b(output.flatten(2))
