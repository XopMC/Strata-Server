"""Validate real typed370 receipts. Never infer counters from flags."""
ADAPTIVE=False
def need(v,m):
 if not v:raise RuntimeError('whole370 receipt: '+m)
def validate_receipt(r,*,bank,completed=False,closed=False):
 w=r['whole_raw'];need(len(w)==45,'authoritative299 StatCount45')
 need((w[0],w[1],w[2],w[3])==(370,139,bank['owner_generation'],bank['format_epoch']),'actual bank generation/format')
 need((r['physical'],r['active'],r['spares'],r['static_entries'])==(1078,1070,8,0),'production fixed capacity')
 need(w[4]==0 and w[5]==r['active'] and not w[8] and w[9]==21470088264 and w[11]==20<<30,'per-rank real requested peak/budget')
 need(w[7]==21463452744,'actual native403 final requested bytes')
 need(not any((r['observer_failures'],r['failed'],w[20],w[21],w[22],w[23])),'no abort/failure/live ticket/borrow/startupscratch')
 need(w[35]==(0 if closed else 6635520),'actual adaptive370 persistent inverse scratch;147 remains absent')
 need(w[17]==w[18]==w[19],'submitted finished released exact')
 need(w[36]==0 and w[37]==1,'GPU1-only qualified startup prime')
 if completed:
  need(r['armed']==1,'real owner armed after warmup')
  need(r['observer_calls']==1280 and r['completed_forwards']==32,'finite32 successful40-layer training census')
  need(w[17]>0 and w[27]>0 and w[39]>0,'real whole GPU1 submitted/routes/completions')
  from fixed_binding_v512 import check
  if check()['mode']==0:need(not any(r[k]for k in('started','upload_completed','published','pending','stages_inflight','H2D_bytes')),'fixed-plan OFF no upload/publication')
 if closed:need(r['closed']==1 and r['positive_drains']>=1 and not any(r[k]for k in('pending','stages_inflight','pinned_bytes')),'actual terminal drain')
 return r
