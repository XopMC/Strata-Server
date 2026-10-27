"""Fixed static403 capacity; real samples are not an absolute driver-peak proof."""
GIB=20
PHYSICAL=1078
BUDGET=GIB<<30
FINAL_BYTES=PHYSICAL*(19906560+16)+4163816
PEAK_BYTES=FINAL_BYTES+6635520
RESERVE=512<<20
def sample(torch,label):
 free,total=map(int,torch.cuda.mem_get_info(1))
 row=dict(label=label,logical_device=1,free_bytes=free,total_bytes=total,Torch_allocated=int(torch.cuda.memory_allocated(1)),Torch_reserved=int(torch.cuda.memory_reserved(1)),reserve_bytes=RESERVE,scope='actual synchronous API sample after startup prime / native4warmup; not absolute CUDA-driver transient peak')
 print(__import__('json').dumps(dict(actual_static403_memory=row)),flush=True)
 return row

def validate_memory(row):
 if row['free_bytes']<RESERVE:raise RuntimeError('403 actual GPU1 postprime/pre-generation reserve below512MiB')
 return row
