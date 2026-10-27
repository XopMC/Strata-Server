"""Pinned installed-library links only; project artifacts remain nonsymlink files."""
from pathlib import Path
import hashlib,os,re
ROOTS=tuple(Path(x)for x in('/usr/local/cuda-13.4','/usr/local/cuda','/usr/lib','/usr/lib64','/lib','/lib64'))
_EVIDENCE={}
def need(v,m):
 if not v:raise RuntimeError('512 R2 pin: '+m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _stat(s):return {k:int(getattr(s,k))for k in('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')}
def _library(p):return bool(re.fullmatch(r'.+\.so(?:\.[0-9]+)*|.+\.a',p.name))and any(p.is_relative_to(root)for root in ROOTS)
def _links(p):
 found={};visited=set()
 def visit(q):
  q=Path(os.path.abspath(q))
  if str(q)in visited:return
  visited.add(str(q));need(len(visited)<=64,'bounded symlink chain')
  for part in reversed((q,*q.parents)):
   if part.is_symlink():
    before=_stat(part.lstat());target=os.readlink(part);need(before==_stat(part.lstat()),'link changed while reading '+str(part))
    found[str(part)]=dict(path=str(part),link_text=target,lstat=before)
    visit(Path(target)if Path(target).is_absolute()else part.parent/target)
 visit(p)
 return [found[k]for k in sorted(found)]
def snapshot(p):
 p=Path(p);need(p.is_absolute()and p.is_file(),'absolute existing file '+str(p))
 if not _library(p):
  need(not p.is_symlink(),'project/non-library artifact must not be symlink '+str(p));return None
 target=p.resolve(strict=True);need(target.is_file()and not target.is_symlink(),'resolved installed-library target')
 need(any(target.is_relative_to(root.resolve())for root in ROOTS),'installed-library target escaped system roots')
 links=_links(p)
 if not links:return None
 return dict(path=str(p),resolved_target=str(target),target_stat=_stat(target.stat()),links=links,scope='installed system library only; exact link chain + resolved target + content/stat')
def verify_pin(r):
 p=Path(r['path']);before=snapshot(p);actual=sha(p);after=snapshot(p)
 need(before==after,'pin target/link changed while hashing '+str(p));need(actual==r['sha256']and('bytes'not in r or p.stat().st_size==r['bytes']),'content/size changed '+str(p))
 if 'installed_system_link'in r:need(before==r['installed_system_link'],'pinned installed-library link/target/stat changed '+str(p))
 if before is not None:
  if str(p)in _EVIDENCE:need(_EVIDENCE[str(p)]==before,'installed-library identity changed in this Root action')
  _EVIDENCE[str(p)]=before
 return before

def make_row(p):
 p=Path(p).absolute();before=snapshot(p);r=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p));need(before==snapshot(p),'pin changed while reading '+str(p))
 if before is not None:
  if str(p)in _EVIDENCE:need(_EVIDENCE[str(p)]==before,'installed-library identity changed in this Root action')
  r['installed_system_link']=before;_EVIDENCE[str(p)]=before
 return r

def decorate(r):
 r=dict(r)
 if r['path']in _EVIDENCE:r['installed_system_link']=_EVIDENCE[r['path']]
 return r

def link_evidence():return [_EVIDENCE[k]for k in sorted(_EVIDENCE)]
