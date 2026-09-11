"""Wait, validate, record C1 repeats and C1–C8; stop on a failed check."""
import argparse,json,subprocess,sys,time,urllib.request
from datetime import datetime
from pathlib import Path

root=Path(__file__).resolve().parent
live=root.parent/'results'/'live'; live.mkdir(parents=True,exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--label',required=True);p.add_argument('--parity-host');p.add_argument('--container');p.add_argument('--reuse-correctness');a=p.parse_args()
run=live/('suite-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+a.label);run.mkdir()
manifest={'url':a.url,'label':a.label,'status':'waiting','steps':{}}
def save(): (run/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
def command(name,args):
    print('START',name,flush=True);last=''
    with (run/(name+'.log')).open('w') as log:
        proc=subprocess.Popen([sys.executable,str(root/args[0]),*args[1:]],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            log.write(line);log.flush();last=line.strip()
            try:
                j=json.loads(line)
                print(name,json.dumps({k:j[k] for k in ['name','category','repeat','automated_check_passed','passed','prompt_tokens','ttft_s','decode_tps','decode_tok_s','count_correct'] if k in j}),flush=True)
            except (ValueError,TypeError): print(name,last,flush=True)
        code=proc.wait()
    manifest['steps'][name]={'exit_code':code,'last_line':last};save()
    if code: raise RuntimeError(name+' failed; see '+str(run/(name+'.log')))
    return Path(last)
def ssh(cmd):
    subprocess.run(['ssh','-o','ConnectTimeout=8',a.parity_host,cmd],check=True,timeout=60)
try:
    start=time.monotonic();save();lastprint=-30
    while time.monotonic()-start<1800:
        try:
            with urllib.request.urlopen(a.url+'/health',timeout=4) as r:
                if r.status==200:break
        except Exception: pass
        elapsed=time.monotonic()-start
        if elapsed-lastprint>=30:print('WAIT',round(elapsed),a.url,flush=True);lastprint=elapsed
        time.sleep(10)
    else: raise TimeoutError('Service not ready within 30 minutes')
    manifest['status']='validating';save()
    d=Path(a.reuse_correctness) if a.reuse_correctness else command('correctness',['check_model_correctness.py',a.url,'--label',a.label])
    manifest['correctness_source']=str(d);save()
    checks=json.loads((d/'results.json').read_text())
    if not all(r['automated_check_passed'] for r in checks):raise RuntimeError('Functional check failed')
    if a.parity_host:
        if not a.container:raise ValueError('--container required with --parity-host')
        for attempt in range(10):
            try:
                with urllib.request.urlopen(a.url+'/flush_cache',timeout=30) as r: r.read()
                break
            except urllib.error.HTTPError:
                if attempt==9:raise
                time.sleep(2)
        ssh('docker exec '+a.container+' touch /tmp/dsv41-prefill-compare')
        try:
            command('actual-prefill-parity',['bench_c1_reference.py',a.url,'--label',a.label+'-parity','--categories','summary'])
        finally: ssh('docker exec '+a.container+' rm -f /tmp/dsv41-prefill-compare')
        logs=subprocess.check_output(['ssh',a.parity_host,'docker logs '+a.container+' 2>&1'],text=True,timeout=60)
        (run/'actual-prefill-parity.log').write_text('\n'.join(x for x in logs.splitlines() if 'PREFILL_PARITY' in x))
    d=command('retrieval',['check_prefill_retrieval.py',a.url,'--label',a.label])
    if not all(r['passed'] for r in json.loads((d/'results.json').read_text())):raise RuntimeError('Retrieval check failed')
    manifest['status']='benchmarking';save()
    command('c1-repeat',['bench_c1_reference.py',a.url,'--label',a.label,'--repeat','3'])
    command('matrix',['bench_matrix.py','--base',a.url+'/v1','--label',a.label,'--out',str(live/'matrix'),'--categories','coding,prose','--warm-prefill','--notes',a.label+'; TP=EP=node count, PP1, DSpark5, decode graphs BS1-8, local Engram, context/KV 8192.'])
    manifest['status']='completed';save();print('COMPLETE',run,flush=True)
except BaseException as e:
    manifest['status']='failed';manifest['error']=repr(e);save();raise
