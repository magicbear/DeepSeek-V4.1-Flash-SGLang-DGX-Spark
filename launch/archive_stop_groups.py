"""Archive logs/configuration before stopping only the named experiment groups."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import argparse,json,subprocess,shlex

p=argparse.ArgumentParser(); p.add_argument('--groups',nargs='+',choices=['a','b'],required=True); a=p.parse_args()
run=Path(__file__).resolve().parent.parent/'results'/'live'/('archived-'+datetime.now().strftime('%Y%m%d-%H%M%S'));run.mkdir(parents=True)
tasks=[]
for g in a.groups:
    name='dsv41-spark4-eager-20260911' if g=='a' else 'dsv41-spark4-cloneall-20260911'
    tasks += [(f'192.168.32.{n}',name) for n in ([99,100,101,102] if g=='a' else [98,103,104,105])]
def ssh(host,cmd,timeout=60):
    r=subprocess.run(['ssh','-o','ConnectTimeout=8',host,cmd],capture_output=True,text=True,timeout=timeout)
    if r.returncode: raise RuntimeError(f'{host} {cmd}: {r.stderr}')
    return r.stdout
def one(task):
    host,name=task
    # Do not store full inspect environment; only experiment resource settings and command.
    info=ssh(host,'docker inspect --format '+shlex.quote('{{json .State}}\n{{json .Config.Cmd}}\n{{json .HostConfig.Memory}}\n{{json .HostConfig.NanoCpus}}')+' '+name)
    logs=ssh(host,'docker logs '+name+' 2>&1')
    (run/f'{host}-{name}.log').write_text(logs)
    (run/f'{host}-{name}.config.txt').write_text(info)
    result=ssh(host,'docker stop --time 15 '+name)
    print(host,result.strip(),flush=True)
    return {'host':host,'name':name,'stopped':True}
with ThreadPoolExecutor(max_workers=8) as pool: result=list(pool.map(one,tasks))
(run/'summary.json').write_text(json.dumps(result,indent=2));print(run,flush=True)
