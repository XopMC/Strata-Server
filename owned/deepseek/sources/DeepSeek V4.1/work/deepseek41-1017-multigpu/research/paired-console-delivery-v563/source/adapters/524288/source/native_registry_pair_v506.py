"""Explicit248/252 native registry over one truthful selected RAM owner."""
import ctypes as C
import json
from pathlib import Path
import numpy as np
from native_cpp_adapter import Region,View,CALLBACK,P,S,I,ROOT
from one_bank_store import OneBankStore as SelectedRegionStore
from one_bank_store import EXPERT
from pair_format_v506 import binding,descriptor,Descriptor


class NativeBundleRegistry:
    def __init__(self,store,library,*,_allow_tiny_fixture=False):
        if not isinstance(store,SelectedRegionStore):raise ValueError('Actual246 selected RAM owner required')
        store.validate_selected(deep=True,allow_fixture=_allow_tiny_fixture)
        self.bank_format=binding(store)
        self.store=store;self.source=store.source;self.registry=self.checkpoint=self.rows=None
        self.refs=0;self.arrays=[];self.payload_copied_bytes=0;self._allow_tiny_fixture=_allow_tiny_fixture
        self.library=Path(library).resolve(strict=True);self.lib=C.CDLL(str(self.library))
        specs={'ds41_native_adapter_error':([],C.c_char_p),
            'ds41_registry_create':([P,S,P,CALLBACK,CALLBACK],P),'ds41_registry_add_segmented':([P,C.POINTER(View),P,S],I),
            'ds145_registry_add_pair32':([P,C.c_char_p,C.POINTER(Descriptor)],I),
            'ds145_registry_pair32_info':([P,P,S],I),
            'ds145_registry_pair32_upload_info':([P,P,S],I),
            'ds41_registry_output16_info_v248':([P,P,S],I),'ds41_registry_destroy':([P],None),
            'ds41_checkpoint_create':([P,C.c_char_p],P),'ds41_checkpoint_destroy':([P],None),
            'ds41_checkpoint_protect_rows':([P,C.c_char_p,C.c_char_p,C.c_char_p,C.c_char_p],I),
            'ds41_rows_create':([P],P),'ds41_rows_destroy':([P],None),
            'ds41_rows_protect':([P,C.c_char_p,C.c_char_p,C.c_char_p,C.c_char_p],I),
            'ds41_rows_gather':([P,C.c_char_p,P,S,P,S],I)}
        for name,(args,result) in specs.items():getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=result
        self.retain=CALLBACK(lambda _:self.change(1));self.release=CALLBACK(lambda _:self.change(-1))
        self.protections=[]
        try:
            regions=[Region(r['address'],r['bytes']) for r in store._regions]
            regions.extend(Region(v['address'],v['bytes']) for v in store.cpu_blocked_cache.values())
            owned=(Region*len(regions))(*regions)
            self.registry=self.check(self.lib.ds41_registry_create(owned,len(owned),id(self),self.retain,self.release),True)
            encoding={'BF16':0,'F32':1,'F16':2,'F8_E4M3':3,'F8_E8M0':4,'I8':5,'U8':6,'I16':7,'I32':8,'I64':9}
            for name,record in store.index.items():
                if name in store.cpu_blocked_cache:
                    v=store.cpu_blocked_cache[name]
                    d=descriptor(store,name,v,bank=self.bank_format)
                    self.check(self.lib.ds145_registry_add_pair32(self.registry,name.encode(),C.byref(d)))
                    continue
                if EXPERT.fullmatch(name):continue  # no omitted main-expert scale/alias/raw descriptor
                codec=record.get('codec')
                if codec=='internal-row-scales':continue
                if codec=='deepseek41-signed-uniform-bf16-group-v1':dtype=11
                elif codec in(None,'heretic-source-bytes-v1'):dtype=encoding[record['dtype']]
                else:raise ValueError('Unknown retained tensor codec')
                if record['dtype']=='I8' and '.experts.' in name and name.endswith('.weight'):dtype=10
                parts=store.retained_tensor_parts(name);self.arrays.extend(parts)
                spans=(Region*len(parts))(*(Region(np.frombuffer(p,np.uint8).ctypes.data,p.nbytes) for p in parts))
                shape=np.asarray(record['shape'],np.int64)
                view=View(name.encode(),dtype,shape.ctypes.data,shape.size,None,record['bytes'],record.get('bits',0),bool(record.get('protected_native_rows')))
                self.check(self.lib.ds41_registry_add_segmented(self.registry,C.byref(view),spans,len(parts)))
                if record.get('protected_native_rows'):
                    protection=record['protected_native_rows']
                    self.protections.append(tuple(v.encode() for v in (name,protection['ids'],protection['weight'],protection['scale'])))
            if binding(store)!=self.bank_format:raise RuntimeError('paired139 format changed during registry145 registration')
            expected=sum(v['bytes'] for v in store.cpu_blocked_cache.values());info=self.pair32_info()
            if info!=(145,139,len(store.cpu_blocked_cache),expected,self.bank_format['owner_generation'],self.bank_format['format_epoch'],1,0):raise ValueError('Actual typed145 registry accounting differs')
            if self.output16_info()!=(1,0,0):raise ValueError('paired139 must not masquerade as old248 entries')
            self.rows=self.check(self.lib.ds41_rows_create(self.registry),True)
            for values in self.protections:self.check(self.lib.ds41_rows_protect(self.rows,*values))
            self.config=json.loads((self.source/'runtime-config.json').read_text())
        except BaseException:
            self.close();raise

    def change(self,amount):
        self.refs+=amount
        if amount>0:self.store.retain_native()
        else:self.store.release_native()

    def check(self,value,pointer=False):
        if (not value) if pointer else bool(value):raise ValueError(self.lib.ds41_native_adapter_error().decode())
        return value

    def output16_info(self):
        if not self.registry:raise ValueError('Selected registry is closed')
        values=(C.c_uint64*3)();self.check(self.lib.ds41_registry_output16_info_v248(self.registry,values,3))
        return tuple(values)

    def pair32_info(self):
        values=(C.c_uint64*8)();self.check(self.lib.ds145_registry_pair32_info(self.registry,values,8));return tuple(values)
    def pair32_upload_info(self):
        values=(C.c_uint64*12)();self.check(self.lib.ds145_registry_pair32_upload_info(self.registry,values,12));return tuple(values)

    def finish_checkpoint(self):
        if self.checkpoint or not self.registry:raise ValueError('Unfinished live selected registry required')
        keys=[line.split()[0] for line in (ROOT/'native-ds41/src/ds41/tests/fixtures/native-geometry.txt').read_text().splitlines()]
        geometry=''.join(k+' '+(' '.join(map(str,self.config[k])) if isinstance(self.config[k],list) else str(self.config[k]))+'\n' for k in keys)
        self.checkpoint=self.check(self.lib.ds41_checkpoint_create(self.registry,geometry.encode()),True)
        for values in self.protections:self.check(self.lib.ds41_checkpoint_protect_rows(self.checkpoint,*values))
        return self

    def gather(self,name,ids,width):
        if not self.rows:raise ValueError('Selected row registry is closed')
        ids=np.ascontiguousarray(ids,dtype=np.int64)
        if ids.ndim!=1 or not len(ids) or type(width)is not int or width<1 or width>65536 or len(ids)*width>32*1024**2:
            raise ValueError('Invalid bounded selected row request')
        out=np.empty((len(ids),width),np.uint16)
        self.check(self.lib.ds41_rows_gather(self.rows,name.encode(),ids.ctypes.data,ids.size,out.ctypes.data,out.size))
        return out

    def close(self):
        if self.checkpoint:self.lib.ds41_checkpoint_destroy(self.checkpoint);self.checkpoint=None
        if self.rows:self.lib.ds41_rows_destroy(self.rows);self.rows=None
        if self.registry:self.lib.ds41_registry_destroy(self.registry);self.registry=None
        if self.refs:raise RuntimeError('Native engines/graphs still retain selected allocations')
        self.arrays.clear();self.store=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
