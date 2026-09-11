"""Install storage adapter in every serving worker, only when explicitly enabled."""
import importlib.abc
import importlib.machinery
import os
import sys

class EngramLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == 'sglang.srt.layers.engram':
            from engram_backend import install
            install(module)
        elif module.__name__ == 'sglang.srt.layers.attention.dsv4.metadata':
            # V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel.
            # SM120 needs its split-128 planner even when the legacy FP8
            # indexer uses the torch path. The upstream guard misses this case.
            cls = module.PagedIndexerMetadata
            original = cls.__post_init__
            def post_init(self):
                if module._IS_SM120 and self.compress_ratio in (1, 2):
                    self.force_deep_gemm_metadata = True
                original(self)
            cls.__post_init__ = post_init
        elif os.getenv('DSV41_PREFILL_TRITON') == '1':
            from sglang.kernels.ops.attention.flash_mla_sm120_triton import flash_mla_sparse_decode_triton
            import logging
            original_prefill=module._flash_mla_sm120_prefill
            # Preserve the FlashInfer <=64-token decode path. Only replace
            # the >64-token route, with the original packed KV page layout.
            seen=set()
            def prefill(*args,**kwargs):
                q=args[0] if args else kwargs['q']
                shape=tuple(q.shape)
                if shape not in seen:
                    seen.add(shape)
                    logging.getLogger(__name__).info('Hybrid sparse MLA: Triton for query shape %s',shape)
                result=flash_mla_sparse_decode_triton(*args,**kwargs)
                if os.path.exists('/tmp/dsv41-prefill-compare'):
                    # Explicitly armed diagnostic; never used for timing runs.
                    import torch
                    fi=original_prefill(*args,**kwargs)[0].float()
                    ref=result[0].float()
                    diff=fi-ref
                    rel=(diff.square().mean()/ref.square().mean().clamp_min(1e-30)).sqrt().item()
                    last=(diff[-1].square().mean()/ref[-1].square().mean().clamp_min(1e-30)).sqrt().item()
                    logging.getLogger(__name__).warning('PREFILL_PARITY shape=%s rel_rmse=%.8f last_row_rmse=%.8f max_abs=%.8f',shape,rel,last,diff.abs().max().item())
                return result
            module._flash_mla_sm120_prefill = prefill

class EngramFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('sglang.srt.layers.engram',
                            'sglang.srt.layers.attention.dsv4.metadata',
                            'sglang.kernels.ops.attention.flash_mla_sm120'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = EngramLoader(spec.loader)
        return spec

if os.environ.get('DSV41_SOURCE'):
    sys.meta_path.insert(0, EngramFinder())
