"""CPU-only byte parity for the exact Engram reader and rank ownership mask."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import random
import struct

p=argparse.ArgumentParser(); p.add_argument('--rank',type=int,required=True); p.add_argument('--tp',type=int,required=True)
a=p.parse_args(); root=Path('/models/dsv41')
idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
lib=C.CDLL('/tmp/dsv41-adapter/librow_store.so')
P=C.c_void_p; U=C.c_uint64
lib.row_store_open.argtypes=[C.c_char_p,U,U,U,U]; lib.row_store_open.restype=P
lib.row_store_range.argtypes=[P,U,U]; lib.row_store_lookup.argtypes=[P]; lib.row_store_close.argtypes=[P]
class Work(C.Structure):
    _fields_=[('store',P),('ids',P),('weights',P),('scales',P),('count',U)]
checks=0; evidence=[]
for layer in [1,14]:
    key=f'layers.{layer}.engram.embed.'; filename=root/idx[key+'weight']
    fd=os.open(filename,os.O_RDONLY)
    n=struct.unpack('<Q',os.pread(fd,8,0))[0]; header=json.loads(os.pread(fd,n,8))
    w=header[key+'weight']; s=header[key+'scale']; rows=w['shape'][0]
    lo=rows*a.rank//a.tp; hi=rows*(a.rank+1)//a.tp
    wo=8+n+w['data_offsets'][0]; so=8+n+s['data_offsets'][0]
    store=lib.row_store_open(str(filename).encode(),rows,wo,so,1024*1024)
    if not store: raise RuntimeError('row_store_open failed')
    lib.row_store_range(store,lo,hi)
    rng=random.Random(1000*layer+a.rank)
    ids=[lo,lo+1,hi-1]+[rng.randrange(lo,hi) for _ in range(29)]
    if lo: ids.append(lo-1)
    if hi<rows: ids.append(hi)
    ids_array=(C.c_int64*len(ids))(*ids)
    weights=(C.c_uint8*(256*len(ids)))(); scales=(C.c_uint8*(8*len(ids)))()
    work=Work(store,C.addressof(ids_array),C.addressof(weights),C.addressof(scales),len(ids))
    for repeat in range(2):
        lib.row_store_lookup(C.byref(work))
        for i,row in enumerate(ids):
            expected_w=os.pread(fd,256,wo+row*256) if lo<=row<hi else bytes(256)
            expected_s=os.pread(fd,8,so+row*8) if lo<=row<hi else bytes(8)
            assert bytes(weights[i*256:(i+1)*256])==expected_w, (layer,row,'weight')
            assert bytes(scales[i*8:(i+1)*8])==expected_s, (layer,row,'scale')
            checks+=1
    evidence.append({'layer':layer,'lo':lo,'hi':hi,'row_checks':len(ids)*2})
    lib.row_store_close(store); os.close(fd)
print(json.dumps({'rank':a.rank,'tp':a.tp,'row_checks':checks,'mismatches':0,'layers':evidence}),flush=True)
