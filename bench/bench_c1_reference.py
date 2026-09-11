"""C1 reference prompts, usage-based token counts, raw SSE and split timings."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.request

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('url'); p.add_argument('--label',required=True)
p.add_argument('--categories',nargs='+',default=['ceiling_count','coding','prose'])
p.add_argument('--repeat',type=int,default=1)
a=p.parse_args()
prompts=json.loads((HERE/'prompts-v1.json').read_text())
byname={x['name']:x for x in prompts['categories']+[prompts['ceiling']]}
out=HERE.parent/'results'/'live'/'bench'; out.mkdir(parents=True,exist_ok=True)
run=out/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+a.label); run.mkdir()
summary=[]
for repeat in range(a.repeat):
    for category in a.categories:
        spec=byname[category]
        payload={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':spec['prompt']}],
                 'max_tokens':spec['max_tokens'],'temperature':0,'stream':True,
                 'stream_options':{'include_usage':True},'chat_template_kwargs':{'thinking':False}}
        result={'url':a.url,'label':a.label,'category':category,'repeat':repeat,'payload':payload,'events':[]}
        t0=time.perf_counter(); first=None; last=None; text=''; usage=None
        result['started_perf_s']=t0
        result['started_at']=datetime.now().isoformat()
        try:
            req=urllib.request.Request(a.url+'/v1/chat/completions',json.dumps(payload).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=300) as response:
                for raw in response:
                    if not raw.startswith(b'data: '): continue
                    line=raw[6:].decode().strip()
                    if line=='[DONE]': break
                    item=json.loads(line); elapsed=time.perf_counter()-t0
                    result['events'].append({'elapsed_s':elapsed,'data':item})
                    if item.get('usage'): usage=item['usage']
                    for choice in item.get('choices',[]):
                        delta=choice.get('delta',{}); content=delta.get('content') or ''
                        if content or delta.get('reasoning_content'):
                            if first is None: first=elapsed
                            last=elapsed
                        text+=content
                        if choice.get('finish_reason') is not None:
                            last=elapsed  # include EOS, which is counted by usage
            wall=time.perf_counter()-t0
            assert usage and first is not None and last>first
            result.update({'text':text,'usage':usage,'ttft_s':first,'decode_elapsed_s':last-first,
                           'decode_tps':(usage['completion_tokens']-1)/(last-first),
                           'e2e_s':wall,'e2e_output_tps':usage['completion_tokens']/wall,
                           'metric_note':'Usage token count; subtract one first token as in reference chat benchmark. SSE events preserved; chunk count is never token count.'})
            if category=='ceiling_count':
                import re
                numbers=[int(x) for x in re.findall(r'\d+',text)]
                result['count_correct']=numbers==list(range(1,81))
        except Exception as exc:
            result['error']=repr(exc)
            if hasattr(exc,'read'): result['error_body']=exc.read().decode(errors='replace')
        path=run/f'{category}-{repeat}.json'; path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        compact={k:v for k,v in result.items() if k not in ('events','payload')}; compact['artifact']=str(path)
        summary.append(compact)
        (run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in compact.items() if k!='text'},ensure_ascii=False),flush=True)
        if 'error' in result: raise SystemExit('Request failed; stopping this benchmark')
print(run,flush=True)
