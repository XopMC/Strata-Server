"""Resolve current NVIDIA indices0/1 afresh; no saved UUID allowlist."""
import csv,io,re,subprocess
INDICES=(0,1)
def parse_rows(text):
    rows={}
    for r in csv.reader(io.StringIO(text)):
        if not r: continue
        if len(r)!=3: raise ValueError('Expected index,UUID,PCI row')
        index=int(r[0].strip());uuid=r[1].strip();pci=r[2].strip()
        if index not in INDICES or index in rows: raise ValueError('Unexpected or duplicate GPU index')
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',uuid): raise ValueError('Malformed runtime device identity')
        if not re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',pci): raise ValueError('Malformed PCI identity')
        rows[index]={'index':index,'uuid':uuid,'pci_bus_id':pci}
    if set(rows)!=set(INDICES)or len({x['uuid']for x in rows.values()})!=2: raise ValueError('Need two distinct GPUs at indices0/1')
    return tuple(rows[i]for i in INDICES)
def resolve_current_pair():
    # Identity tokens are ephemeral CUDA routing/locking handles, NOT a saved hardware requirement.
    # Resolving NVML indices avoids CUDA_DEVICE_ORDER/FASTEST_FIRST reordering choosing another card.
    text=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True,timeout=15)
    return parse_rows(text)
