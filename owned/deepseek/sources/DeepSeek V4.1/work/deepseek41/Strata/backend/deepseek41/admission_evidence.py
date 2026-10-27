"""Find a completed, request-bound audit without depending on its retry number."""
import hashlib
import json
from pathlib import Path
import re


def completed_revision(root,prefix,accept):
    root=Path(root)
    if not re.fullmatch(r'[a-z0-9-]+-v',prefix):raise ValueError('Invalid project audit prefix')
    output=root.parents[1]/'outputs/deepseek41/resident-lab'
    candidates=[]
    for path in output.glob(prefix+'*.json'):
        match=re.fullmatch(re.escape(prefix)+r'([1-9][0-9]*)\.json',path.name)
        if match:candidates.append((int(match[1]),path))
    if len(candidates)>100:raise ValueError('Too many audit revisions')
    for _,path in sorted(candidates,reverse=True):
        if path.is_symlink()or path.stat().st_size>16*1024**2 or path.with_suffix('.error.json').exists():continue
        try:
            result=json.loads(path.read_text())
            request_path=root/'state/resident-requests'/(path.stem+'.json')
            if request_path.is_symlink()or request_path.stat().st_size>1024**2:continue
            request=json.loads(request_path.read_text())
        except (OSError,ValueError):continue
        digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if(result.get('id')!=path.stem or request.get('id')!=path.stem or
           result.get('request_sha256')!=digest or result.get('request_kind')!=request.get('kind')or
           type(result.get('resident_owner_pid'))is not int or result['resident_owner_pid']<1 or
           result.get('all_payload_reads_from_ram')is not True):continue
        if accept(result):return result
    raise ValueError('No completed matching source audit: '+prefix)
