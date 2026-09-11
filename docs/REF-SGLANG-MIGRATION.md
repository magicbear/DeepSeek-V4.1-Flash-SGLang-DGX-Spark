# DeepSeek-V4.1-Flash：Spark vLLM 配方对 SGLang 的参考价值

参考仓库：[tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/ca662ac35193c69ace9cee37f13a94abf2eff0fc)，固定 commit ca662ac35193c69ace9cee37f13a94abf2eff0fc。这是 4 台 Spark 的 vLLM 实现，不能把其补丁直接作为 SGLang 插件使用。

| 参考点 | 当前 SGLang 实施与边界 |
|---|---|
| Engram 按需读取原始磁盘行 | 原 native C++ callback 已实现，保留全局 row ID 和 TP 所有权 |
| 每 rank 的本地 NVMe 稀疏副本 | 已实现并分发，TP4 每节点47.21GiB，TP8每节点23.60GiB；稀疏副本只给 reader，完整模型 loader仍读BeeGFS |
| 行偏移和边界校验 | 原文件CPU字节校验536项；每本地副本8008项；graph replay多长度逐位校验通过 |
| DSpark k=5 + 精确batch图捕获 | SGLang block5对应verify6，捕获BS1–8；4机 target/draft graph成功并真实使用 |
| 限制编译并行，持久化JIT | MAX_JOBS=2、FLASHINFER_NVCC_THREADS=1；节点本地/root/.cache映射 |
| SM121 稀疏注意力/索引器页大小修复 | vLLM和SGLang内存布局不同；SGLang采用Triton prefill / FlashInfer decode混合路径，4机已通过基础文本和独立检索检查，不能直接照搬vLLM补丁 |
| GPU快慢状态探针 | 本集群也测得间歇慢态，频率读数相同；对历史8机扩展差是线索，尚无因果对照 |

## 当前结果的限制

4机全FlashInfer优化版短计数C1曾测得60.00–72.72decode TPS，代码48.44–59.41，正文19.20–28.02。但扩大到完整任务集出现无关摘要/表格等错误响应，矩阵已停止；这些速度不能作为完整正确服务的验收。原始响应均保存。扩大对照后，eager也出现摘要异常；全Triton注意力+DSpark+graphs通过9类基础检查，计数44–46、代码约36、正文约17 TPS。

4机混合路径已通过9类基础检查、9个输入长度样本、1502/2922/5914-token随机记录码检索，以及三任务C1–C8共108条请求的基本内容检查。无标签C1三次中位decode：计数71.04、代码57.92、正文27.37 TPS；C8总输出吞吐分别284.16/207.87/87.81 tokens/s。2950/5853-token输入的预热后TTFT为11.502/23.141秒，有效prefill为256.5/252.9 tokens/s。8机同配置也已全部通过：C1中位计数78.32、代码67.03、正文28.58 TPS；有效prefill 417.7/444.0 tokens/s；C8总吞吐计数293.84、代码216.19、正文84.21 tokens/s。另一次6轮连续计数复测最高86.15、中位85.40 TPS，与交错三任务协议分开。

真实摘要输入的rank0逐层数值对照中，FlashInfer与Triton prefill相对RMSE约0.6%–1.0%；未发现非有限值。混合路径改善了这些已测任务，但上述对照并未证明全FlashInfer错误输出的具体根因。

仓库中92.2TPS是特定计数场景，不是各种自然语言任务的一般性能。本文不把外部vLLM成绩当作我们SGLang的实测，也不包含RTX性能。
