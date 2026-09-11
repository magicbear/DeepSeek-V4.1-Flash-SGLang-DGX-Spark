"""Six count-only C1 repeats with read-only NVML; no concurrent GPU microbenchmark."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse,json,subprocess,threading,time,statistics
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--label',required=True);a=p.parse_args()
out=HERE.parent/'results'/'live'/('jitter-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+a.label);out.mkdir(parents=True)
procs={};threads=[];samples={};ready={}
def start(n):
    ready[n]=threading.Event();samples[n]=[]
    proc=subprocess.Popen(['ssh','-o','ConnectTimeout=8',f'192.168.32.{n}','timeout','90','nvidia-smi','--query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu,clocks_event_reasons.active','--format=csv,noheader,nounits','-lms','200'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    procs[n]=proc
    with (out/f'{n}-nvml.jsonl').open('w') as f:
        for line in proc.stdout:
            fields=[x.strip() for x in line.split(',')];r={'client_perf_s':time.perf_counter(),'line':line.strip()}
            if len(fields)==6:
                try:r.update(sm_mhz=float(fields[1]),power_w=float(fields[2]),utilization=float(fields[3]),temp_c=float(fields[4]),throttle=int(fields[5],16));ready[n].set()
                except ValueError:pass
            f.write(json.dumps(r)+'\n');f.flush();samples[n].append(r)
try:
    for n in range(98,106):
        t=threading.Thread(target=start,args=(n,),daemon=True);t.start();threads.append(t)
    deadline=time.monotonic()+12
    while time.monotonic()<deadline and not (len(ready)==8 and all(v.is_set() for v in ready.values())):time.sleep(.1)
    if len(ready)!=8 or not all(v.is_set() for v in ready.values()):raise RuntimeError('NVML telemetry missing; do not run unobserved jitter test')
    proc=subprocess.run(['python3',str(HERE.parent/'bench'/'bench_c1_reference.py'),a.url,'--label',a.label+'-jitter','--categories','ceiling_count','--repeat','6'],capture_output=True,text=True,timeout=180)
    (out/'bench.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError('C1 jitter benchmark failed')
    ref=Path(proc.stdout.strip().splitlines()[-1]);rows=json.loads((ref/'summary.json').read_text())
finally:
    for proc in procs.values():proc.terminate()
    for t in threads:t.join(timeout=3)
summary=[]
for r in rows:
    start=r['started_perf_s']+r['ttft_s'];end=start+r['decode_elapsed_s'];hoststats={}
    for n,ss in samples.items():
        s=[x for x in ss if start<=x['client_perf_s']<=end and 'sm_mhz' in x]
        if s:hoststats[str(n)]={k:statistics.median(x[k] for x in s) for k in ['sm_mhz','power_w','utilization','temp_c']}|{'throttle_flags':sorted({x['throttle'] for x in s}),'samples':len(s)}
    summary.append({'repeat':r['repeat'],'decode_tps':r['decode_tps'],'count_correct':r['count_correct'],'ttft_s':r['ttft_s'],'hosts':hoststats})
(out/'summary.json').write_text(json.dumps({'benchmark_source':str(ref),'measurements':summary},indent=2));print(json.dumps(summary,indent=2));print(out)
