"""Preserve the source's EXISTING E2M1/E4M3 group16 KV codes without BF16 storage.

This is a storage change, unlike re-quantizing native latents to GGML Q4_0.
The source quantizer is called once on the original post-RoPE latent values.
"""
import hashlib
import json
from pathlib import Path
import torch


class NativeFP4Cache:
    def __init__(self,capacity,width,device,*,batch=1):
        if type(capacity)is not int or capacity<1 or type(width)is not int or width<16 or width%16 or batch!=1:
            raise ValueError('Native FP4 KV requires single-batch group16 aligned rows')
        self.capacity,self.width,self.device=capacity,width,torch.device(device)
        self.codes=torch.zeros(1,capacity,width//2,dtype=torch.uint8,device=self.device)
        self.scales=torch.zeros(1,capacity,width//16,dtype=torch.uint8,device=self.device).view(torch.float8_e4m3fn)
        self.rows_written=0

    @property
    def storage_bytes(self):return self.codes.numel()+self.scales.numel()

    def write(self,first,values,quantizer):
        if(type(first)is not int or first<0 or values.ndim!=3 or values.shape[0]!=1 or values.shape[-1]!=self.width or
           values.device!=self.device or values.dtype!=torch.bfloat16 or first+values.shape[1]>self.capacity):
            raise ValueError('Native FP4 KV write outside capacity or original BF16 latent contract')
        packed,scales=quantizer(values,16,False,scale_dtype=torch.float8_e4m3fn)
        if packed.shape!=(*values.shape[:-1],self.width//2)or scales.shape!=(*values.shape[:-1],self.width//16)or scales.dtype!=torch.float8_e4m3fn:
            raise ValueError('Native FP4 source quantizer changed its code/scale contract')
        if not bool(torch.isfinite(scales.float()).all()):raise ValueError('Nonfinite native KV source scale')
        self.codes[:,first:first+values.shape[1]].copy_(packed.view(torch.uint8))
        self.scales[:,first:first+values.shape[1]].copy_(scales)
        self.rows_written+=values.shape[1]

    def raw_buffers(self,first,last):
        if not 0<=first<=last<=self.capacity:raise ValueError('Native FP4 snapshot outside cache')
        return self.codes[:,first:last],self.scales[:,first:last]

    def prefix(self,length):
        if type(length)is not int or not 0<=length<=self.capacity:raise ValueError('Native FP4 prefix outside capacity')
        return NativeFP4Prefix(self,length)


class NativeFP4Prefix:
    def __init__(self,cache,length):self.cache,self.length=cache,length
    @property
    def shape(self):return(1,self.length,self.cache.width)
    @property
    def dtype(self):return torch.bfloat16
    @property
    def device(self):return self.cache.device


def dequantize(codes,scales):
    if(codes.dtype!=torch.uint8 or scales.dtype!=torch.float8_e4m3fn or codes.shape[:-1]!=scales.shape[:-1]or
       codes.shape[-1]!=scales.shape[-1]*8):raise ValueError('Native FP4/E4 KV layout mismatch')
    ids=torch.stack((codes&15,codes>>4),-1).flatten(-2).long()
    table=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32,device=codes.device)
    values=table[ids].unflatten(-1,(-1,16))*scales.float().unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


def implementation_sha256():
    digest=hashlib.sha256()
    for name in('native_fp4_cache.py','native_fp4_attention.py','admission_evidence.py'):
        digest.update(name.encode());digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def admit_component(reference):
    root=Path(__file__).resolve().parents[3]
    from .admission_evidence import completed_revision
    completed_revision(root,'ram-isolated-native-fp4-kv-v',lambda proof:
        proof.get('memory_errors')==0 and proof.get('child',{}).get('implementation_sha256')==implementation_sha256()and
        len(proof.get('child',{}).get('checks',[]))==18 and
        proof.get('child',{}).get('all_matches_original_bf16')is True and
        proof.get('source_revision')==reference.store.ram_preload_report['source_revision'])


def attention_forward(graph,attention,x,start):
    from .native_fp4_attention import sparse_attn
    batch,length,_=x.shape;frequencies=attention.freqs_cis[start:start+length];rd=attention.rope_head_dim
    qr=attention.q_norm(attention.wq_a(x))
    query=attention.wq_b(qr).unflatten(-1,(attention.n_local_heads,attention.head_dim))
    graph.apply_rotary_emb(query[...,-rd:],frequencies)
    window,indices=attention._window_kv(x,frequencies,start)
    packed,compressed=attention._compress_kv(x,qr,start,window.shape[1])
    if not isinstance(packed,NativeFP4Prefix):raise ValueError('Native FP4 attention lost its packed owner')
    output=sparse_attn(query,window,packed,attention.attn_sink,torch.cat((indices,compressed),-1),attention.softmax_scale)
    graph.apply_rotary_emb(output[...,-rd:],frequencies,True)
    output=output.view(batch,length,attention.n_local_groups,-1)
    weights=attention.wo_a.weight.view(attention.n_local_groups,attention.o_lora_rank,-1)
    output=torch.einsum('bsgd,grd->bsgr',output,weights)
    return attention.wo_b(output.flatten(2))
