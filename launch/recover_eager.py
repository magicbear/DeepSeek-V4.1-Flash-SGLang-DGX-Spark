"""Bounded-memory Spark V4.1 bring-up; preflight all hosts before mutations.

Default: render commands only. --probe checks connectivity. --launch saves
previous logs, stops only this experiment's containers and launches eager mode.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess

import launch_spark_v41 as base

ROOT = Path(__file__).resolve().parent


def ssh(host, cmd, timeout=25):
    return subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
         '-o', 'ConnectionAttempts=1', host, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def probe(host):
    try:
        p = ssh(host, 'hostname; uptime; free -m; timeout 8 docker ps --format "{{.Names}} {{.Status}}"')
        return {'host': host, 'ok': p.returncode == 0,
                'stdout': p.stdout, 'stderr': p.stderr, 'returncode': p.returncode}
    except subprocess.TimeoutExpired:
        return {'host': host, 'ok': False, 'stderr': 'SSH command deadline exceeded'}


def command(size, rank, dspark=False, name=None, force_copy=False, graphs=False,
            concurrency=1, local_engram=False, mla_backend=None):
    tokens = shlex.split(base.command(size, rank, 8100 + size))
    tokens[tokens.index('--name') + 1] = name or f'dsv41-spark{size}-eager-20260911'
    inner = tokens[-1]
    inner = inner.replace('--mem-fraction-static ' + ('0.70' if size == 4 else '0.82'),
                          '--mem-fraction-static 0.85 --max-total-tokens 8192')
    inner = inner.replace('--context-length 409600', '--context-length 8192')
    inner = inner.replace('--chunked-prefill-size 2048', '--chunked-prefill-size 1024')
    inner = inner.replace('--max-running-requests 4 --cuda-graph-max-bs-decode 4',
                          '--max-running-requests 1 --cuda-graph-backend-decode disabled '
                          '--cuda-graph-backend-prefill disabled --disable-flashinfer-autotune --skip-server-warmup')
    if not dspark:
        inner = inner.replace('--speculative-algorithm DSPARK --speculative-dspark-block-size 5 ', '')
    inner = inner.replace('--max-running-requests 1 ', f'--max-running-requests {concurrency} ')
    if graphs:
        graph_bs = ' '.join(str(x) for x in range(1, concurrency + 1))
        inner = inner.replace('--cuda-graph-backend-decode disabled',
                              '--cuda-graph-backend-decode full --cuda-graph-bs-decode ' + graph_bs)
    # Bound the lifetime of CPU weight views. Queuing every expert copy holds
    # many mmap-backed shards while GB10 already owns most physical RAM.
    inner = inner.replace('--load-format safetensors',
                          '--load-format safetensors --model-loader-extra-config ' +
                          shlex.quote('{"num_threads":1}'))
    patch = ("sed -i 's/use_async_loading = should_async_load(loaded_weight)/"
             "use_async_loading = False  # Spark bounded weight staging/' "
             "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py; ")
    inner = inner.replace('export PYTHONPATH=', patch + 'export PYTHONPATH=')
    if force_copy:
        patch = ("sed -i 's/if not needs_copy:/if False and not needs_copy:/' "
                 "/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py; ")
        inner = inner.replace('export PYTHONPATH=', patch + 'export PYTHONPATH=')
    tokens[-1] = inner
    tokens[tokens.index('DSV41_CACHE_GIB=32')] = 'DSV41_CACHE_GIB=8'
    # Persist expensive JIT results on each host. Avoid putting compiler caches
    # on BeeGFS. cgroup limits help bound CPU/host-memory use, but are not a
    # guarantee of a hard limit on all CUDA allocations.
    index = tokens.index(base.IMAGE)
    tokens[index:index] = [
        '--memory=100g', '--memory-swap=100g', '--cpus=12',
        '-v', '/data/dsv41-spark-cache:/root/.cache',
        '-e', 'OMP_NUM_THREADS=4', '-e', 'MAX_JOBS=2',
        '-e', 'FLASHINFER_NVCC_THREADS=1',
        '-e', 'SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1',
    ]
    if local_engram:
        index = tokens.index(base.IMAGE)
        tokens[index:index] = ['-e','DSV41_ENGRAM_LOCAL=/engram-local',
                              '-v',f'/data/dsv41-engram-local/tp{size}-rank{rank}:/engram-local:ro']
    if mla_backend:
        index=tokens.index(base.IMAGE)
        tokens[index:index]=['-e', 'SGLANG_SM120_FLASHMLA_BACKEND='+('flashinfer' if mla_backend=='hybrid' else mla_backend)]
        if mla_backend=='hybrid':
            index=tokens.index(base.IMAGE)
            tokens[index:index]=['-e','DSV41_PREFILL_TRITON=1']
    return shlex.join(tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, choices=[4, 8], default=8)
    p.add_argument('--dspark', action='store_true')
    p.add_argument('--name', help='Container name for an isolated comparison')
    p.add_argument('--force-copy', action='store_true', help='Clone all CPU expert weights, including complete tensors')
    p.add_argument('--graphs', action='store_true')
    p.add_argument('--concurrency', type=int, choices=range(1,9), default=1)
    p.add_argument('--local-engram', action='store_true')
    p.add_argument('--mla-backend', choices=['flashinfer','triton','torch','hybrid'])
    p.add_argument('--adapter', help='Shared adapter source directory')
    p.add_argument('--hosts', nargs='+', help='Explicit management addresses, rank order')
    p.add_argument('--master', help='Rendezvous address reachable by all ranks')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--probe', action='store_true')
    action.add_argument('--launch', action='store_true')
    args = p.parse_args()
    if args.adapter:
        base.ADAPTER = args.adapter
    if args.hosts:
        if len(args.hosts) != args.size or len(set(args.hosts)) != args.size:
            p.error('--hosts must contain exactly --size distinct hosts')
        base.HOSTS = args.hosts
        base.MASTER = args.master or args.hosts[0]
    elif args.master:
        base.MASTER = args.master
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run = ROOT.parent / 'results' / 'live' / stamp
    run.mkdir(parents=True)
    hosts = base.HOSTS[:args.size]
    cmds = {host: command(args.size, rank, args.dspark, args.name, args.force_copy,
                          args.graphs,args.concurrency,args.local_engram,args.mla_backend) for rank, host in enumerate(hosts)}
    (run / 'config.json').write_text(json.dumps(vars(args), indent=2))
    for rank, host in enumerate(hosts):
        (run / f'rank{rank}.sh').write_text(cmds[host] + '\n')
    print(f'Artifacts: {run}', flush=True)
    if not (args.probe or args.launch):
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, hosts))
    (run / 'preflight.json').write_text(json.dumps(results, indent=2))
    for result in results:
        print(json.dumps(result), flush=True)
    if not all(result['ok'] for result in results):
        raise SystemExit('Preflight failed. No remote changes made.')
    if not args.launch:
        return
    names = [args.name] if args.name else [f'dsv41-spark{args.size}-20260910', f'dsv41-spark{args.size}-eager-20260911']
    # Preserve containers and logs, including failed attempts, before relaunch.
    for host in hosts:
        for name in names:
            exists = ssh(host, f'docker inspect --format ' + shlex.quote('{{.State.Status}}') + f' {name}')
            if exists.returncode:
                continue
            logs = ssh(host, f'docker logs {name} 2>&1', timeout=45)
            if logs.returncode:
                raise RuntimeError(f'Cannot preserve logs: {host} {name}')
            (run / f'{host}-{name}.log').write_text(logs.stdout)
            stopped = ssh(host, f'docker stop --time 15 {name}', timeout=40)
            if stopped.returncode:
                raise RuntimeError(stopped.stderr)
            renamed = ssh(host, f'docker rename {name} {name}-saved-{stamp}')
            if renamed.returncode:
                raise RuntimeError(renamed.stderr)
    for host in hosts[1:] + hosts[:1]:
        result = ssh(host, cmds[host], timeout=60)
        if result.returncode:
            raise RuntimeError(f'{host}: {result.stderr}')
        print(host, result.stdout.strip(), flush=True)


if __name__ == '__main__':
    main()
