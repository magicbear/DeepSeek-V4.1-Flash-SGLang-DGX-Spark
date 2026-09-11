# DeepSeek-V4.1-Flash / SGLang / DGX Spark 复现

固定参考：[tonyd2wild Spark vLLM配方](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc)。本文命令用于现有Spark集群，不需要docker pull。模型、ARM64镜像和编译源码已存在；其它机器需先满足同样前置条件。

## 固定环境

- 主机192.168.32.98–105，各1个GB10 GPU；主节点rendezvous地址192.168.32.184。
- 镜像lmsysorg/sglang:dev-v4f-2dgx-v2，ID `67873eb93b994736ab534111f79b5aa93d2575b973ebf9d276c40d548ff9afec`。
- 模型`/mnt/beegfs/models/DeepSeek-V4.1-Flash`；overlay为`/mnt/beegfs/sglang-v41-python`、`sglang-v41-csrc`、`sglang-v41-jitinclude`。
- 混合适配器`/mnt/beegfs/dsv41-spark-adapter-hybrid`；本工作区源码在`adapter-local-engram/`。启动时复制到容器，不热修改运行中实例。
- 本地Engram：每主机`/data/dsv41-engram-local/tpN-rankR`，带已校验`verified.json`；仅Engram reader使用稀疏副本。
- JIT缓存`/data/dsv41-spark-cache`映射到容器`/root/.cache`。

## 启动

从仓库根目录运行。启动器默认只生成命令，添加`--launch`才执行；每轮保存配置、rank命令、旧实例日志。切换4/8机前，需停止占用相同节点的本实验组；启动器仅处理指定同名容器，不会停止其它名字。

四机（本轮B组）：

```bash
python3 experiments/spark-dsv41-20260910/recover_eager.py \
  --size 4 --hosts 192.168.32.98 192.168.32.103 192.168.32.104 192.168.32.105 \
  --master 192.168.32.184 --name dsv41-spark4-cloneall-20260911 \
  --force-copy --dspark --graphs --concurrency 8 --local-engram \
  --adapter /mnt/beegfs/dsv41-spark-adapter-hybrid --mla-backend hybrid --launch
```

八机：

```bash
python3 experiments/spark-dsv41-20260910/recover_eager.py \
  --size 8 --master 192.168.32.184 --name dsv41-spark8-hybrid-20260911 \
  --force-copy --dspark --graphs --concurrency 8 --local-engram \
  --adapter /mnt/beegfs/dsv41-spark-adapter-hybrid --mla-backend hybrid --launch
```

主要参数：TP=EP=设备数，PP1；context/KV8192；chunked prefill1024；最大并发8；DSpark block5；decode full graph BS1–8；prefill graph关闭；autotune关闭；跳过自动多模态warmup；顺序权重加载，所有CPU专家权重clone；CPU12核、host memory100GiB（不覆盖全部GPU分配）。

## 验证与测量

```bash
python3 experiments/spark-dsv41-20260910/finish_spark_bench.py \
  http://192.168.32.98:8108 --label spark8-hybrid
```

四机使用8104端口，八机8108。脚本等待健康，再依次执行基础正确性、独立记录码检索、C1三次复测、代码/正文/计数C1–C8、两档有效prefill。失败停止并保存原始响应和manifest。用于性能比较前还需查看输出；基本自动检查不等同于完整质量评估。

`--parity-host`诊断会在指定rank上临时开启FlashInfer/Triton双算对比；默认关闭，不用于计时成绩。临时flag是容器`/tmp/dsv41-prefill-compare`，脚本在finally中移除。

不要把SSE块数当token数。性能脚本使用服务端usage；decode、TTFT、总输出吞吐分别保存。2-token短答不作稳定decode排名。发布原始JSON位于`results/sglang-final/`，新测试输出位于`results/live/`。
