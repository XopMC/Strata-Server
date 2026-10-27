"""Explicit inherited layout139 capability; no inference from old metadata."""
import ctypes as C,os,re
from pathlib import Path
KEYS={'schema','layout_tag','format_epoch','owner_generation','owner_epoch','inventory_sha256'}
NAME=re.compile(r'layers\.([0-9]|[1-3][0-9])\.ffn\.experts\.([0-9]|[1-9][0-9]|[12][0-9][0-9]|3[0-7][0-9]|38[0-3])\.(w[123])\.weight')
class Descriptor(C.Structure):
 _fields_=[('struct_bytes',C.c_uint64),('layout_tag',C.c_uint32),('bits',C.c_int32),('data',C.c_void_p),('bytes',C.c_uint64)]+[(n,C.c_int32)for n in ('rows','columns','layer','expert','role','reserved')]+[('owner_generation',C.c_uint64),('bank_format_epoch',C.c_uint64)]
assert C.sizeof(Descriptor)==72
def binding(store):
 value=getattr(store,'bank_format_binding_v146',None)
 if value is None or set(value)!=KEYS:raise RuntimeError('explicit controller146 paired capability required')
 b=dict(value);b['owner_epoch']=dict(b['owner_epoch'])
 if b['schema']!='deepseek1017-bank-format/146' or b['layout_tag']!=139 or type(b['format_epoch'])is not int or b['format_epoch']<2 or b['owner_generation']!=store.generation or b['owner_epoch']!=store._owner_epoch or b['owner_epoch']['pid']==os.getpid():raise RuntimeError('exact inherited paired139 parent/epoch differs')
 pid=b['owner_epoch']['pid'];stat=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
 if int(stat[19])!=b['owner_epoch']['start_ticks'] or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=b['owner_epoch']['boot_id']:raise RuntimeError('paired139 original owner no longer live')
 if type(b['inventory_sha256'])is not str or len(b['inventory_sha256'])!=64 or any(c not in '0123456789abcdef'for c in b['inventory_sha256']):raise RuntimeError('paired139 inventory identity absent')
 return b
def descriptor(store,name,r,*,bank=None):
 # Parent146 cannot relayout while this inherited child exists. Validate one
 # snapshot per phase and recheck at its end, not /proc twice per matrix.
 b=binding(store)if bank is None else bank
 match=NAME.fullmatch(name)
 if match is None:raise RuntimeError('canonical paired139 descriptor name required')
 layer,expert,which=match.groups();layer=int(layer);expert=int(expert);role={'w1':0,'w3':1,'w2':2}[which]
 rows,columns=(5120,2304)if role==2 else(2304,5120)
 if any(type(r[k])is not int for k in('bytes','bits','address','rows','columns'))or r['bytes']!=6635520 or r['bits']!=0 or r['address']<=0 or r['address']%2097152 or(r['rows'],r['columns'])!=(rows,columns):raise RuntimeError('actual paired139 shape/bits/allocation differ')
 return Descriptor(C.sizeof(Descriptor),139,0,r['address'],r['bytes'],rows,columns,layer,expert,role,0,b['owner_generation'],b['format_epoch'])
