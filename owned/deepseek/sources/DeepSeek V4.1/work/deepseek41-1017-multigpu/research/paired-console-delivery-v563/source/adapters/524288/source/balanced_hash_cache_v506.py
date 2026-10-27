"""Only432 link-input digests: once per PID, immutable identity checks thereafter.
No CUDA/native import, environment mutation or file writes. Never caches a refusal.
"""
import hashlib,os,threading
from pathlib import Path
_LOCK=threading.RLock()
_PID=os.getpid()
_CACHE=None
_FAILED=False

def _after_fork():
 global _LOCK,_PID,_CACHE,_FAILED
 _LOCK=threading.RLock();_PID=os.getpid();_CACHE=None;_FAILED=False
if hasattr(os,'register_at_fork'):os.register_at_fork(after_in_child=_after_fork)

def _need(v,m):
 if not v:raise RuntimeError('436 balanced input identity: '+m)
def _identity(p):
 p=Path(p);_need(p.is_absolute()and not p.is_symlink(),'canonical non-symlink file '+str(p))
 s=p.stat();_need(p.is_file(),'regular file '+str(p))
 return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def _digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _small(p):
 before=_identity(p);sha=_digest(p);_need(before==_identity(p),'file changed during digest '+str(p));return(before,sha)
def _verify(rows,policy,author_pins,seal):
 global _CACHE
 # Policy is already checked against literal flags/selectedDSO by bulk binding.
 # The small author manifest and exact seal bytes stay checked on every call.
 rows=tuple((r['path'],r['sha256'],r['bytes'])for r in rows)
 _need(len(rows)==len(set(x[0]for x in rows)),'duplicate inputs')
 author=_small(author_pins);seal_path=Path(seal)
 seal_now=_small(seal_path)if seal_path.exists()else None
 key=(rows,policy,str(author_pins),author,str(seal_path))
 if _CACHE is None:
  identities=[]
  for name,sha,size in rows:
   before=_identity(name);_need(before[2]==size,'declared input extent '+name)
   _need(_digest(name)==sha,'full input SHA '+name)
   _need(before==_identity(name),'input changed during full digest '+name)
   identities.append(before)
  _CACHE=(key,tuple(identities),seal_now)
  return
 old_key,identities,old_seal=_CACHE
 _need(key==old_key,'immutable manifest/policy/author binding changed')
 for (name,sha,size),expected in zip(rows,identities):
  _need(_identity(name)==expected,'dev/ino/size/mtime_ns/ctime_ns changed '+name)
 if old_seal is not None:_need(seal_now==old_seal,'sealed packet changed or disappeared')
 elif seal_now is not None:
  # Root metadata --bind/--seal can create the FIRST seal in this same PID.
  # Runtime enters with a seal already present; removal/change is always refused.
  _CACHE=(old_key,identities,seal_now)

def verify_once(rows,*,policy,author_pins,seal):
 global _PID,_CACHE,_FAILED
 # Also handle PID mismatch when tests/fork environments bypass at_fork hooks.
 if _PID!=os.getpid():_after_fork()
 with _LOCK:
  _need(not _FAILED,'prior identity failure; no rehash/fallback in this PID')
  try:_verify(rows,policy,author_pins,seal)
  except BaseException:
   _FAILED=True
   raise
