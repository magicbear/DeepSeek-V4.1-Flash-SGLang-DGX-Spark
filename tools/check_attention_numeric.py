"""Packed FP8 sparse attention vs explicit dequantized PyTorch reference.

Run as an isolated process inside the runtime image, not inside the scheduler.
"""
import json,math,torch,os
from sglang.kernels.ops.attention import flash_mla_sm120 as m
torch.manual_seed(42)
dev='cuda'
def packed(pbs,pages):
    stride=math.ceil(pbs*584/576)*576
    raw=torch.zeros((pages,stride),dtype=torch.uint8,device=dev)
    data=torch.randn((pages,pbs,512),device=dev,dtype=torch.bfloat16)*.2
    n=data[:,:,:448].to(torch.float8_e4m3fn)
    data[:,:,:448]=n.to(torch.bfloat16)
    body=raw[:,:pbs*576].view(pages,pbs,576)
    body[:,:,:448]=n.view(torch.uint8)
    body[:,:,448:]=data[:,:,448:].contiguous().view(torch.uint8)
    raw[:,pbs*576:pbs*584]=127
    return raw.as_strided((pages,pbs,1,584),(stride,584,584,1)),data.reshape(-1,512)

cache,kv=packed(256,4)
for T in [32,64,65,74,292]:
  for extra_pbs in [0,128,64]:
    H=int(os.getenv('CHECK_HEADS','32'))
    q=torch.randn((T,1,H,512),dtype=torch.bfloat16,device=dev)
    lengths=torch.randint(1,200,(T,),dtype=torch.int32,device=dev)
    idx=torch.randint(0,1024,(T,128),dtype=torch.int32,device=dev)
    lengths=lengths.clamp(max=128)
    idx.masked_fill_(torch.arange(128,device=dev)[None,:]>=lengths[:,None],-1)
    sink=torch.randn(H,device=dev)
    kw=dict(q=q,k_cache=cache,indices=idx[:,None,:],topk_length=lengths,
            attn_sink=sink,head_dim_v=512,softmax_scale=512**-.5)
    if extra_pbs:
      ec,ek=packed(extra_pbs,8)
      ei=torch.randint(0,extra_pbs*8,(T,128),dtype=torch.int32,device=dev)
      el=torch.randint(0,128,(T,),dtype=torch.int32,device=dev)
      ei.masked_fill_(torch.arange(128,device=dev)[None,:]>=el[:,None],-1)
      kw.update(extra_k_cache=ec,extra_indices_in_kvcache=ei[:,None,:],extra_topk_length=el)
    # Reference uses the same known quantized values, without page conversion.
    gathered=kv[idx.clamp(min=0).long()].float(); valid=idx>=0
    if extra_pbs:
      gathered=torch.cat([gathered,ek[ei.clamp(min=0).long()].float()],dim=1)
      valid=torch.cat([valid,ei>=0],dim=1)
    score=torch.einsum('thd,tkd->thk',q[:,0].float(),gathered)*512**-.5
    score.masked_fill_(~valid[:,None,:],float('-inf'))
    prob=torch.softmax(torch.cat([score,sink[None,:,None].expand(T,-1,1)],dim=-1),dim=-1)[:,:,:-1]
    ref=torch.einsum('thk,tkd->thd',prob,gathered)
    try:
      actual=m.flash_mla_with_kvcache_sm120(**kw)[0][:,0].float()
      torch.cuda.synchronize()
      diff=actual-ref
      row={'T':T,'heads':H,'extra_pbs':extra_pbs,'relative_rmse':(diff.square().mean()/ref.square().mean()).sqrt().item(),
           'max_abs':diff.abs().max().item(),'finite':bool(actual.isfinite().all())}
    except Exception as e: row={'T':T,'extra_pbs':extra_pbs,'error':repr(e)}
    print(json.dumps(row),flush=True)
