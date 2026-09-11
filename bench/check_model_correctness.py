"""Functional text checks plus a prompt-length sweep; saves full SSE."""
import argparse,ast,json,re
from pathlib import Path
from datetime import datetime
import bench_matrix as bench

p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--label',required=True);p.add_argument('--sweep-only',action='store_true');a=p.parse_args()
out=Path(__file__).resolve().parent.parent/'results'/'live'/('correctness-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+a.label);out.mkdir(parents=True)
bench.RAW_DIR=out/'raw';bench.RAW_DIR.mkdir();rows=[]
def check(name,prompt,limit,validator):
    r=bench.stream_chat(a.url+'/v1','deepseek-v4.1-flash',prompt,limit,timeout=300)
    try: ok=bool(validator(r['text']))
    except Exception: ok=False
    r.update(name=name,automated_check_passed=ok)
    rows.append(r);(out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k!='raw_file'},ensure_ascii=False),flush=True)
if not a.sweep_only:
  for name,prompt,limit in bench.CATEGORIES+[bench.CEILING]:
    validators={
      'coding':lambda t:any(isinstance(x,ast.FunctionDef) and x.name=='merge_intervals' for x in ast.walk(ast.parse(re.search(r'```(?:python)?\s*([\s\S]*?)```',t).group(1)))),
      'json':lambda t:set(['name','city','founded','genres','staff'])<=set(json.loads(t)) and len(json.loads(t)['genres'])==5 and len(json.loads(t)['staff'])==3,
      'math':lambda t:bool(re.search(r'14[:：]06|2[:：]06\s*(?:p\.?m\.?|PM)',t)),
      'format':lambda t:all(x in t.lower() for x in ['apples','bread','milk','eggs','8.95','|']),
      'summary':lambda t:len(re.findall(r'^\s*[-*•]\s',t,re.M))==3 and any(x in t.lower() for x in ['batter','energy','storage']),
      'reasoning':lambda t:'Ben' in t and 'dog' in t and ('cannot' in t.lower() or 'two' in t.lower() or 'either' in t.lower()),
      'prose':lambda t:'heat' in t.lower() and any(x in t.lower() for x in ['compressor','refrigerant']),
      'narrative':lambda t:'bottle' in t.lower() and any(x in t.lower() for x in ['light','keeper']),
      'ceiling_count':lambda t:[int(x) for x in re.findall(r'\d+',t)]==list(range(1,81)),
    }
    check(name,prompt,512 if name in ['coding','math','reasoning'] else 300,validators[name])
for repeats in [0,8,16,24,32,48,64,128,256]:
    prompt=('Background: '+'blue river stone. '*repeats+'\n' if repeats else '')+'Question: What is the capital of France? Reply with only the city name.'
    check('length_sweep_'+str(repeats),prompt,32,lambda t:t.strip().rstrip('.').lower()=='paris')
print(out,flush=True)
