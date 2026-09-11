"""Cold unique-prefix retrieval checks before accepting prefill throughput."""
import argparse,json
from datetime import datetime
from pathlib import Path
import bench_matrix as bench
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--label',required=True);a=p.parse_args()
out=Path(__file__).resolve().parent.parent/'results'/'live'/('retrieval-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+a.label);out.mkdir(parents=True)
bench.RAW_DIR=out/'raw';bench.RAW_DIR.mkdir();rows=[]
for target in [1000,2000,4000]:
  key='RAVEN-'+str(70193+target*7)
  words=bench.filler(int(target*.55),target+9057).split()
  words.insert(len(words)//3,f'\nThe record code is {key}.\n')
  prompt=f'[retrieval case {target}] Read the following stored notes and find the record code.\n'+' '.join(words)+'\nWhat is the record code? Reply with only that exact code.'
  r=bench.stream_chat(a.url+'/v1','deepseek-v4.1-flash',prompt,32,timeout=600)
  r.update(target=target,expected=key,passed=r['text'].strip()==key)
  rows.append(r);(out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
  print(json.dumps(r,ensure_ascii=False),flush=True)
print(out,flush=True)
