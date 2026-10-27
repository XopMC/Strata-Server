"""Installed-library dependency pins only. Project/source/profile pins stay strict."""
import hashlib,os,stat
from pathlib import Path
from content_cache_v565 import CONTENT_HASH_CACHE

INSTALLED_LIBRARY_ROOTS=(Path('/usr/local/cuda-13.4/targets/x86_64-linux/lib'),Path('/usr/lib/x86_64-linux-gnu'),Path('/lib/x86_64-linux-gnu'))
def need(ok,text):
 if not ok:raise ValueError(text)
def stat_record(s):return dict(device=s.st_dev,inode=s.st_ino,mode=s.st_mode,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,nlink=s.st_nlink)
def allowed(p):return any(p.is_relative_to(root)for root in INSTALLED_LIBRARY_ROOTS)
def links(p):
 out=[];current=p
 for _ in range(16):
  if not current.is_symlink():return out
  s=current.lstat();target=os.readlink(current);out.append(dict(path=str(current),link=target,stat=stat_record(s)))
  current=Path(os.path.abspath(current.parent/target))if not Path(target).is_absolute()else Path(target)
  need(allowed(current),'installed link chain escapes explicit system-library roots')
 raise ValueError('installed library link chain exceeds16')
def verify_installed_dependency(row):
 """Called only for an explicit DEPENDENCY_PINS row failing canonical no-link."""
 p=Path(row['path']);need(p.is_absolute()and allowed(p),'only explicit installed CUDA/system-library dependencies may use links')
 resolved=p.resolve(strict=True);need(allowed(resolved)and resolved.is_file()and not resolved.is_symlink(),'installed dependency target must be canonical regular system library')
 before=stat_record(resolved.stat());chain=links(p)
 need(stat.S_ISREG(before['mode'])and before['bytes']==row['bytes'],'installed dependency exact pinned bytes')
 # Preserve this family's exact root/link/evidence policy on EVERY call. Only
 # the canonical target's known content digest is shared with bindings.pin.
 verified=CONTENT_HASH_CACHE.verify_project(dict(path=str(resolved),bytes=row['bytes'],sha256=row['sha256']))
 opened=verified.evidence['target_stat'];digest=verified.evidence['sha256']
 need(opened==before==stat_record(resolved.stat())and p.resolve(strict=True)==resolved and links(p)==chain,'installed dependency changed while verifying')
 need(digest==row['sha256'],'installed dependency exact content SHA')
 return dict(path=str(p),resolved_target=str(resolved),bytes=before['bytes'],sha256=digest,link_chain=chain,target_stat=before)
