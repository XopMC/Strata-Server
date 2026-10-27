def flags():return {'DS1017_WHOLE_DOWN_METADATA_V391':'1'}
def need(v,m):
 if not v:raise RuntimeError('static403 transport: '+m)
def validate(a,whole,closed=False,startup=False):
 mode=int(flags()['DS1017_WHOLE_DOWN_METADATA_V391']);need(len(a)==16 and a[:2]==[391,16]and not any(a[14:]),'actual391 counters/errors')
 need(a[3]==1 and a[4+mode]==1 and a[5-mode]==0 and a[2]==int(not closed),'actual single production Operator mode/lifetime')
 n=a[8]+a[9];need(n>0 and a[8+mode]==n and a[9-mode]==0 and a[7]==(a[6]if mode else 0),'actual selected DOWN branch')
 need(a[10]==a[13]==n*(640 if mode else 176896)and a[11]==n*176896 and a[12]==a[11]-a[10],'actual complete transport bytes')
 w=whole['whole_raw'];need(n==w[43]+1 and a[6]==w[39]+1,'actual chunks/enqueues plus startup')
 if startup:
  need(not closed and a[6]==n==1 and not any(w[i]for i in(17,18,19,39,43)),'actual constructorprime only, no realwholejob')
  need(w[29]==0,'Owner H2D remains0 beforefirstrealfinish; Operatorprime is separatelycounted')
 else:need(w[29]==a[6]*736+n*(65920+(640 if mode else 176896)),'actual Owner H2D accounting afterrealfinish')

def delta(before,after):
 mode=int(flags()['DS1017_WHOLE_DOWN_METADATA_V391']);need(before[:2]==after[:2]==[391,16],'actual391 deltaABI');d=[b-a for a,b in zip(before,after)]
 need(all(v>=0 for v in d[3:])and not d[14]and not d[15],'actual391 monotonic/errors')
 n=d[8]+d[9];need(d[6]>0 and n>0 and d[8+mode]==n and d[9-mode]==0 and d[7]==(d[6]if mode else 0),'actual391 generation selected branch')
 need(d[10]==d[13]==n*(640 if mode else 176896)and d[11]==n*176896 and d[12]==d[11]-d[10],'actual391 delta byteledger');return d

def read(lib):
 import ctypes as C
 out=(C.c_uint64*16)();f=lib.ds391_whole_down_transport_info;f.argtypes=[C.c_void_p,C.c_int];f.restype=C.c_int
 rc=int(f(out,16))
 if rc:raise RuntimeError('actual391 getter refused rc='+str(rc))
 return list(map(int,out))
