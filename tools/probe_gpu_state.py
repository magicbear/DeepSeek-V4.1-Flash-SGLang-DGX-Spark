"""Run the reference's CPU-independent GPU state probe on idle serving nodes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import subprocess
import argparse

here=Path(__file__).resolve().parent
source=(here/'gpuflip.py').read_text()
parser=argparse.ArgumentParser(); parser.add_argument('--hosts',nargs='+',type=int,default=list(range(98,106)))
parser.add_argument('--telemetry',action='store_true'); args=parser.parse_args()
run=here.parent/'results'/'live'/('gpu-state-'+datetime.now().strftime('%Y%m%d-%H%M%S')); run.mkdir(parents=True)
def one(n):
    host=f'192.168.32.{n}'
    name='dsv41-spark4-eager-20260911' if n in [99,100,101,102] else 'dsv41-spark4-cloneall-20260911'
    print('PROBE',host,flush=True)
    telemetry=None; telemetry_out=None
    try:
        if args.telemetry:
            telemetry_out=(run/f'{host}-nvml.csv').open('w')
            telemetry=subprocess.Popen(['ssh','-o','ConnectTimeout=8',host,'timeout','100','nvidia-smi',
                    '--query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu,clocks_event_reasons.active',
                    '--format=csv,noheader,nounits','-lms','200'],stdout=telemetry_out,stderr=subprocess.STDOUT)
        p=subprocess.run(['ssh','-o','ConnectTimeout=8',host,'docker','exec','-i',name,'python3','-'],input=source,text=True,capture_output=True,timeout=240)
        (run/f'{host}.log').write_text(p.stdout+p.stderr)
        result={'host':host,'returncode':p.returncode,'blocks':[x for x in p.stdout.splitlines() if x.startswith(('BLOCK','START','END'))]}
    except Exception as exc: result={'host':host,'error':str(exc)}
    finally:
        if telemetry:
            telemetry.terminate(); telemetry.wait(timeout=15); telemetry_out.close()
    print(json.dumps(result),flush=True)
    return result
with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(one,args.hosts))
(run/'summary.json').write_text(json.dumps(results,indent=2)); print(run,flush=True)
