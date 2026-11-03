# 双 H200：从空环境到首轮实验

已确认每卡约 140GB 显存、磁盘充足；**仅 CPU 区能联网，GPU 区离线，两区共享目录**。
当前服务器请先按 [CPU 联网准备 / GPU 离线运行指南](OFFLINE.md) 完成环境和资源准备，再从本页第 3 节继续。
下列第 1–2 节仅适用于 GPU 节点本身也能联网的其他环境，不适用于当前服务器。
CPU 内存、驱动和 CUDA toolkit 尚未提供；预检会输出，启动脚本不改系统驱动。
每个进程独占一张卡，模型不跨卡；本版不需要 torchrun/DeepSpeed/vLLM。

## 1. 环境

要求 Linux、Python 3.11 或 3.12、Git、可用 NVIDIA 驱动。
默认安装 PyTorch 2.10.0/torchvision 0.25.0 的 CUDA 12.8 wheel。
若服务器驱动不支持，先调整服务器环境；不要在不匹配的环境里启动训练。
causal-conv1d 如无匹配的预编译 wheel，需要 nvcc、C++ 编译工具与兼容 CUDA toolkit。

```bash
nvidia-smi
free -h
df -h .
nvcc --version
git clone https://github.com/Vannico233/DiG.git
cd DiG
bash scripts/bootstrap_h200.sh
source .venv/bin/activate
```

安装在仓库自己的 `.venv` 中，避免污染系统训练环境。可用 `PYTHON_BIN=python3.11` 指定解释器。
`MAX_JOBS=4` 限制 CUDA 扩展编译并发；机器 RAM 较小时再降低。
主要 HF/Torch 版本固定在 `requirements/h200-constraints.txt`；HF 接口经过微型随机 Qwen CPU 检查，CUDA 组合仍须服务器验证。
Python 包的完整解析结果写入 `environment.lock.txt`，实验时保留该文件，不在一次实验中升级包。
后续相同环境复现可安装该 lock；CUDA 扩展仍需匹配相同 Torch/CUDA。

若 FLA/causal-conv1d 安装或导入失败，先修复环境。
默认配置禁止悄悄使用极慢 fallback；不要把缺失快速算子的运行当作效率实验。

## 2. 下载与准备

```bash
export HF_HOME="$PWD/assets/hf-cache"
export TOKENIZERS_PARALLELISM=false
python -m dig_rl.prepare --assets assets --config configs/qwen35_4b.json
```

脚本从官方 Qwen model repo、openai/gsm8k 和项目沿用的 MATH numeric repo 下载。
首次运行先写 `assets/download_lock.json`，将 main 解析为不可变 revision，再下载。
网络中断后重跑同一条命令，会沿用已锁定版本及 HF 缓存。
已有数据会核对文件摘要与分割配置，不静默覆盖成另一份实验数据。
若想换 H、分割大小或 revision，请使用新的 assets 路径，并在后续命令中一致指定。
不需要把权重、数据或 token 上传到 GitHub。

生成：

```text
assets/model/                 权重、tokenizer、processor、dig_model.json
assets/data/gsm8k/            train/calibration/audit/validation/test.jsonl
assets/data/math_numeric/     同上，另有 manifest.json / excluded.json
```

## 3. 预检和 32 题流程检查

```bash
python -m dig_rl.launch --stage preflight --output runs/pilot01
python -m dig_rl.launch --stage smoke --output runs/pilot01
```

预检逐卡验证模型加载、BF16、快速算子、LoRA 反传及短回答生成。
其中 synthetic reward 仅用于保证 RL 反向路径被执行，不代表模型效果。
smoke 在 32 道校准题构成的池上跑候选批次，使用明确标注的 a=1/rho=.5 检查负系数与更新路径；
不会保存给正式实验使用的模型，也不会被成本 gate 当作通过证据。
全阶段状态写 `*_status.json`，完整错误写对应 `.log`。

## 4. 冻结模型审计

```bash
python -m dig_rl.launch --stage audit --output runs/pilot01
```

两卡分别审计两个数据集。每套 32 道题拟合 a，另外 96 道题评估；每题两组独立 G=4 回答。
审计仍会实际调用模型，耗时不可当作免费。
优先查看：

```bash
cat runs/pilot01/audit/gsm8k/decision.json
cat runs/pilot01/audit/math_numeric/decision.json
```

关键字段：`enabled`、`a`、`rho`、`tau2/nu2/epsilon2`、`d_seconds/c_seconds`、
`candidates[*].gain_lower_approx`、`audit_seconds`。
不通过是允许的研究结果；不要为了跑出 DiG 结果手工修改 decision.json。
若多题奖励全部为 0 或 1，RLOO 组内信号可能不足；先检查截断、奖励解析和任务难度。

## 5. 双卡对照

```bash
python -m dig_rl.launch --stage train --output runs/pilot01
```

每个数据集 GPU 0 跑 RLOO，GPU 1 跑 DiG，使用同一初始权重、LoRA 初始化和配置。
每卡按队列处理两个数据集，没有额外全量 sweep。
若某数据集 gate 未通过，启动器标记 DiG 为 skipped，避免重复跑完全相同的 fallback。
直接调用 `dig_rl.train --arm dig` 时则会明确执行 `rloo_fallback`。

默认每次训练最多 200 步或 4 小时计费预算，先到者结束。
DiG 的计费从整个 audit 的成本开始，RLOO 为 0；共同下载/模型加载和独立评估不计入训练预算。
如果 audit 已耗尽预算，DiG 不训练。不得将 200 步对比自动解释成等计算量加速。
模型状态变化后 a/rho 可能失效，因此先看短程曲线，再决定是否进行第二状态审计。

检查日志：

```bash
tail -f runs/pilot01/gsm8k_dig.log
cat runs/pilot01/train_status.json
cat runs/pilot01/train/gsm8k_dig/summary.json
```

用 `--datasets gsm8k` 只跑一个数据集；用 `--gpus 2 3` 更换两张卡；`--dry-run` 只显示命令。

## 6. 验证和汇总

```bash
python -m dig_rl.launch --stage evaluate --output runs/pilot01
python -m dig_rl.summarize --run runs/pilot01
```

默认评估 final checkpoint 的 256 道 validation 题，每题 4 次采样。
`results.csv` 包含实际有效方法、步数、计费秒数、含审计的回答/token 数量、成功率与截断率；
`comparisons.json` 为逐题配对差异。缺失/失败/跳过不会被填成零成功率。
冻结最终实验后可显式评估 test：

```bash
CUDA_VISIBLE_DEVICES=0 python -m dig_rl.evaluate \
  --model-dir assets/model --data-dir assets/data/gsm8k \
  --checkpoint runs/pilot01/train/gsm8k_rloo/checkpoint.pt \
  --split test --max-problems 100000 --samples 4 --output runs/pilot01/test/gsm8k_rloo
```

基座评估使用同一命令但省略 `--checkpoint`。修改学习率、H/G/B 或 seed 后，应使用新 run 目录并重新审计。
不同 seed 的 LoRA 初始化也不同，不能直接复用旧 audit 的尺度和 gate。

## 7. 恢复和第二状态审计

启动器不会重复启动已存在的 stage。中断后检查对应 status/log，然后对同一个命令增加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=1 python -m dig_rl.train \
  --model-dir assets/model --data-dir assets/data/gsm8k \
  --arm dig --audit-file runs/pilot01/audit/gsm8k/decision.json \
  --output runs/pilot01/train/gsm8k_dig --resume
```

恢复从最后保存的完整状态开始，回滚该 checkpoint 之后的日志记录。
崩溃后未保存的进度会重算；恢复并不保证跨硬件/算子版本位级相同。
模型、配置、数据、代码或 audit 决策变化时拒绝恢复，避免混合不同实验。

第二模型状态：

```bash
CUDA_VISIBLE_DEVICES=1 python -m dig_rl.audit \
  --model-dir assets/model --data-dir assets/data/gsm8k \
  --initial-checkpoint runs/pilot01/train/gsm8k_rloo/checkpoint.pt \
  --output runs/state02_audit_gsm8k
```

这检验跨状态稳定性；若利用上一轮 audit 结果反复调算法，需要准备新的独立审计题。
此时继续训练要同时传同一 `--initial-checkpoint` 与新 `--audit-file`，不能把第二状态的 a 用于原始基座。

## 已知尚未验证的部分

本地只有 CPU，尚未执行 Qwen3.5-4B/H200 的真实 GPU 流程，不能承诺吞吐或提速。
安装版本组合以服务器 preflight 和 smoke 通过为准。若发生 OOM，先停止并统一降低两臂配置，再重新准备/审计；不要只改变一个方法的设置。
当前直接使用 Transformers 同步生成，未接入 vLLM；这是为了保持采样策略与梯度评分一致，性能优化应另做对照。

官方依据：[Qwen 模型](https://huggingface.co/Qwen/Qwen3.5-4B)、
[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、
[PyTorch 安装版本](https://pytorch.org/get-started/previous-versions/)、
[FLA](https://github.com/fla-org/flash-linear-attention)、
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)。
