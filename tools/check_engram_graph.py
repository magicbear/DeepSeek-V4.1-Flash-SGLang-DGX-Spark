"""Actual native host-callback replay vs original checkpoint bytes."""
import json,os,random,struct,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,'/tmp/dsv41-adapter')
import engram_backend as backend
class Embed(torch.nn.Module):
    def _empty(self,indices):
        return torch.empty((*indices.shape,256),device=indices.device,dtype=torch.bfloat16)
fake=SimpleNamespace(EngramEmbedding=Embed,get_parallel=lambda:SimpleNamespace(tp_size=4,tp_rank=0))
backend.install(fake)
root=Path('/models/dsv41'); index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
for layer in [1,14]:
    key=f'layers.{layer}.engram.embed.'
    fd=os.open(root/index[key+'weight'],os.O_RDONLY)
    n=struct.unpack('<Q',os.pread(fd,8,0))[0]; h=json.loads(os.pread(fd,n,8));rows=h[key+'weight']['shape'][0]
    wo=8+n+h[key+'weight']['data_offsets'][0]; so=8+n+h[key+'scale']['data_offsets'][0]
    embed=Embed(rows,256,layer); rng=random.Random(layer)
    for count in [6,48,64,65,256,292]:
        ids=torch.zeros(count,dtype=torch.int64,device='cuda')
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): embed._owned_rows(ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): out=embed._owned_rows(ids)
        maxdiff=0.
        for repeat in range(3):
            values=[rng.randrange(rows) for _ in range(count)]
            values[0]=0;values[-1]=rows//4-1
            ids.copy_(torch.tensor(values,device='cuda'));graph.replay();torch.cuda.synchronize()
            weights=b''.join(os.pread(fd,256,wo+v*256) if v<rows//4 else bytes(256) for v in values)
            scales=b''.join(os.pread(fd,8,so+v*8) if v<rows//4 else bytes(8) for v in values)
            w=torch.frombuffer(bytearray(weights),dtype=torch.float8_e4m3fn).float().view(count,8,32)
            s=torch.frombuffer(bytearray(scales),dtype=torch.float8_e8m0fnu).float().view(count,8,1)
            expected=(w*s).reshape(count,256).to(torch.bfloat16)
            diff=(out.cpu().float()-expected.float()).abs().max().item();maxdiff=max(maxdiff,diff)
            if not torch.equal(out.cpu(),expected): raise AssertionError((layer,count,repeat,diff))
        print(json.dumps({'layer':layer,'count':count,'replays':3,'max_abs_diff':maxdiff}),flush=True)
    os.close(fd)
