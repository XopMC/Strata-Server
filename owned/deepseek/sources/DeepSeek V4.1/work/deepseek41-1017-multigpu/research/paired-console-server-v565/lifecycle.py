"""Same-process all-or-rollback resource leases and durable close receipts."""
import contextlib,fcntl,json,os,stat
from pathlib import Path

@contextlib.contextmanager
def acquire_leases(paths):
    # Deterministic order, no wait while partially holding resources.
    paths=sorted({str(Path(p).absolute()) for p in paths});held=[]
    try:
        for name in paths:
            path=Path(name);path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
            try:
                st=os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid():raise ValueError('Lease must be an owned regular file: '+name)
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:os.close(fd);raise
            held.append(fd)
        yield tuple(held)
    finally:
        # Never unlink a flock inode: waiters could otherwise lock a new inode.
        for fd in reversed(held):
            try:fcntl.flock(fd,fcntl.LOCK_UN)
            finally:os.close(fd)

def write_exclusive(path,value):
    path=Path(path);payload=(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    except BaseException:
        # A partial receipt is never accepted by parser/identity validation.
        raise

def validate_close_proof(proof,*,parent,child,bank,context,library,library_sha,exit_code):
    if exit_code!=0:raise ValueError('Only normal child exit can establish close')
    expected=dict(schema=252,phase='PAIR_CHILD_CLOSED_ZERO_LEASES',parent=parent,child=child,
        context=context,runtime_library=library,runtime_sha256=library_sha,
        bank_format_binding=bank,native_leases=0,FIFO_closed=True,quality_admitted=False)
    if proof!=expected:raise ValueError('Close receipt identity, lease or lifecycle mismatch')
    if type(proof['native_leases']) is not int:raise ValueError('Literal zero native leases required')
    return proof
