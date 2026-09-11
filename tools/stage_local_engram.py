"""Distribute only each TP rank's exact Engram rows to independent local caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shlex
import struct
import subprocess

here=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('--hosts',nargs='+',required=True); p.add_argument('--mbps',type=int,default=300)
a=p.parse_args(); tp=len(a.hosts); assert tp in (4,8)
source=Path('/mnt/beegfs/models/DeepSeek-V4.1-Flash')
idx=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
counts={}
for layer in [1,14]:
    name=f'layers.{layer}.engram.embed.weight'
    with (source/idx[name]).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; counts[layer]=json.loads(f.read(n))[name]['shape'][0]
code=(here/'stage_engram_rows.py').read_text()
run=here/'recovery'/('local-engram-tp'+str(tp)+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')); run.mkdir()
def one(pair):
    rank,host=pair; dst=f'/data/dsv41-engram-local/tp{tp}-rank{rank}'
    specs=[f'{l}:{n*rank//tp}:{n*(rank+1)//tp}' for l,n in counts.items()]
    required=sum((n*(rank+1)//tp-n*rank//tp)*264 for n in counts.values())
    probe=subprocess.run(['ssh','-o','ConnectTimeout=8',host,'df -B1 --output=avail /data | tail -1'],capture_output=True,text=True,check=True)
    if int(probe.stdout.strip()) < required+20*2**30: raise RuntimeError(f'Insufficient free local space: {host}')
    command=shlex.join(['python3','-',str(source),dst,*specs,f'--mbps={a.mbps}'])
    (run/f'{host}.command.txt').write_text(command+'\n')
    print('STAGE',host,rank,round(required/2**30,2),'GiB',flush=True)
    with (run/f'{host}.log').open('w') as out:
        process=subprocess.Popen(['ssh','-o','ConnectTimeout=8',host,command],stdin=subprocess.PIPE,stdout=out,stderr=subprocess.STDOUT,text=True)
        process.communicate(code,timeout=1800)
    result={'host':host,'rank':rank,'tp':tp,'returncode':process.returncode,'path':dst}
    print(json.dumps(result),flush=True)
    if process.returncode: raise RuntimeError(f'Staging failed: {host}; see {run}')
    return result
with ThreadPoolExecutor(max_workers=tp) as pool: results=list(pool.map(one,enumerate(a.hosts)))
(run/'summary.json').write_text(json.dumps(results,indent=2)); print(run,flush=True)
