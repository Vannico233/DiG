# DiG v1

正确示范梯度预测 + **生成前**随机抽查 + 无偏残差校正。
当前版本使用 `h_i = a u_i`，不加入公共向量、额外 SFT、PPO 裁剪或自适应选题。

```text
L = (1/B) sum_b [a D_i^H + Z_b/rho * (L_RLOO,b - a D_i^H)]
D_i^H = -log pi(demo_i | question_i) / H
A_j = R_j - mean(R_other_responses)
L_RLOO = -sum_j stopgrad(A_j) log pi(response_j | question_i) / (G H)
```

SGD、无动量、无权重衰减、无梯度裁剪；学习率 `eta0 / sqrt(step + 1)`。
固定 `H` 归一化，未选中位置不生成回答，所有位置的损失除以候选总数 `B`。
抽中位置的示范系数可以为负。零抽中批次保留预测更新。`rho=1` 跳过示范计算。

## 已确定的实验设置

- Qwen/Qwen3.5-4B，BF16 基座、FP32 LoRA；只在语言模型线性层添加 LoRA。
- GSM8K 和 MATH numeric；使用数据集标准解答作为单条示范。
- 当前四张 H100（每卡 80GB）使用独立工作进程，四组探索对照见 [四卡实验指南](docs/EXPLORATORY.md)。原双 H200 启动器仍保留，不做 DDP。
- 原始策略采样：temperature=1、top_p=1、top_k=0、thinking=false；固定 boxed 数值奖励。
- 默认 H=2048、G=4、B=8、LoRA r=16/alpha=32；初始 SGD lr=0.01 **只是待验证的 pilot 设置，不是已调优结论**。
- 32 题校准、96 题独立审计、256 题验证，其余合格训练题用于训练；原始 test 单独保留。

## 从零开始

当前服务器是 **CPU 区联网、GPU 区离线、两区共享目录**。请使用 [离线准备指南](docs/OFFLINE.md)，可从现有 Python 3.9 环境启动：

```bash
# CPU 联网区，在共享 DiG 仓库内
git pull --ff-only
bash scripts/prepare_cpu_online.sh

# 准备成功后，进入 GPU 离线区的同一目录
source scripts/activate_offline.sh
python scripts/offline_setup.py gpu
python -m dig_rl.launch --stage preflight --output runs/pilot01
python -m dig_rl.launch --stage smoke --output runs/pilot01
python -m dig_rl.launch --stage audit --output runs/pilot01
# 先阅读 runs/pilot01/audit/*/decision.json，再运行训练阶段。
python -m dig_rl.launch --stage train --output runs/pilot01
python -m dig_rl.launch --stage evaluate --output runs/pilot01
python -m dig_rl.summarize --run runs/pilot01
```

每个阶段独立启动，不会自动接着跑整个实验。任何失败均返回非零状态。
新环境放在 `.runtime/py311/`，不会修改 `/etc/inspire`；两区须使用相同的绝对路径。
GPU 也能联网的独立机器使用 [双 H200 运行指南](docs/SERVER.md) 的在线安装流程。
没有 GPU 的本地验证：`pip install -e '.[test]' && python -m pytest -q`。

已有实验训练收益较弱时，可先运行 [短程训练信号诊断](docs/SIGNAL_DIAGNOSTIC.md)：
四卡比较两个数据集的两档 SGD 学习率，每组仅四步训练与 32 题验证，复用已有初始模型输出，
记录逐组奖励差异、生成/反向耗时、显存及 GPU 利用率。代码在本地实现并同步 GitHub，
CPU 空间同步和准备，GPU 空间由使用者运行；当前实验尚未完成或 GPU 忙碌时拒绝启动新诊断。

## 审计与回退

在同一个冻结模型状态上，先拟合非负尺度 a，再使用独立问题与每题两组独立 RLOO 梯度估计预测误差、题间差异、题内噪声。
不把单次 RL 梯度当真值；不把负的带噪误差估计截成零。
审计记录实际生成/反传耗时，且将整个审计开销按计划候选数摊销进预测成本。

```text
V = tau² + nu²/rho + (1/rho - 1) epsilon²
F_DiG = (d + rho*c) V
F_RL  = c (tau² + nu²)
```

在预先固定的 rho 网格上选择具有正收益余量的候选。
不确定性采用题目级 delete-one jackknife 与多重比较修正的正态近似，
**是保守的工程筛选规则，不是有限样本概率证书**。
审计不通过时，双卡启动器跳过 DiG 训练，仍可运行完整 RLOO。
直接调用 DiG 训练入口则明确记录为 `rloo_fallback`。

## 可复现与范围

- 模型/数据 HF revision 在首次下载前锁定；数据分割、文件校验、LoRA 初始化种子和代码摘要都会记录。
- `checkpoint.pt` 保存全部可训练参数、SGD 状态、独立采样 RNG、步数和成本账本；支持同一版本精确协议恢复。
- 默认单次训练上限为 200 步或 4 小时计费预算，先到者停止。DiG 计入审计准备成本。
  等步数不等于等计算预算；一批计算可能越过时间边界，结果明确记录超额。
- 当前门槛是冻结模型上的估计。a/rho 在短程训练中固定，不能把该门槛当成全程成立的定理。
  第二模型状态需要重新审计。暂不实现训练中自动刷新、共享 v0、历史回答重复更新或多示范来源切换。
- 收敛证明仅针对其假设成立时的一阶平稳性，不保证准确率或实际加速。
- CPU 测试验证代数、梯度路径、数据隔离和恢复流程；**H200 算子、显存、吞吐和真实效果须在服务器预检及实验中确认**。

实现细节见 [算法与审计约定](docs/METHOD.md)。

## 第二轮：批量生成与图文探索实验

新入口为 `scripts/round2.py`，旧入口和结果保留。新数据位于 `assets/round2`：
GSM8K、MATH 原训练集（测试为 MATH-500）、Geo3K、ScienceQA 图片子集。
每个数据集使用同配置 RLOO/DiG，默认 G=8、候选题 B=32、同时生成 16 条回答，
LoRA r16/alpha32、AdamW 1e-6、32 步或每个 worker 两小时先到停止。
GSM8K/Geo3K/ScienceQA 输出上限 4096，MATH 为 8192；损失固定除以 2048。
每个回答有独立随机流；DiG 抽样在生成前完成，不补选问题来填满 batch。

GPU 0/1/2/3 分别处理 GSM8K/MATH/Geo3K/ScienceQA，每卡有 RLOO 和 DiG 两个候选 worker。
先比较单 worker 与双 worker 的每卡总吞吐，再选择每卡 1 或 2 个并发。
双 worker 总吞吐至少提高 15%、两者峰值 reserved 显存之和小于 64 GiB 时采用双任务。
512-token 测速并不保证长序列显存，OOM 会停止当前阶段并保留 checkpoint。
校准阶段每卡单 worker，训练阶段再使用选出的并发；已有外部 GPU 进程会阻止启动。

CPU 页面执行（先同步 Git）：

```bash
bash scripts/prepare_round2.sh
```

Geo3K / ScienceQA 的 Parquet 下载使用固定版本、官方 SHA256 校验和 8 MiB 分块续传，
默认 4 个连接，测速后选择官方（继承代理）、镜像直连或官方直连。
中断后重跑同一命令；已准备的数据集校验后跳过，旧 Hub 缓存只读复用。
分块连续 12 次没有进展才失败；有有效新增字节会继续恢复。
可用 `DIG_DATA_SOURCE=hf-mirror bash scripts/prepare_round2.sh` 固定镜像；
`DIG_DATA_CONNECTIONS=2` 可降低连接数。不要删除 `assets/round2` 或 `assets/hf-cache`。
等待日志出现 `Round-two CPU assets ready` 后再启动 GPU。

GPU 页面执行：

```bash
nohup bash scripts/start_round2.sh runs/round2_20261105 > .runtime/round2.log 2>&1 < /dev/null &
```

监测与结果（先激活离线环境）：

```bash
python scripts/round2.py status --output runs/round2_20261105
python scripts/round2.py summarize --output runs/round2_20261105
tail -n 30 .runtime/round2.log
```

运行依次完成测速、新校准、共享 Base 验证、成对训练和最终验证，训练每步保存。
相同版本恢复使用同一启动命令；协议或代码改变必须使用新目录。
快速验证是训练集内留出的 64 道题、每题一次 greedy，不读取官方 test。
完整官方测试和三种子扩展应在 pilot 配置冻结后另行执行。

这一轮明确是固定 rho=0.5 的探索比较，校准不等于独立审计通过；
AdamW 不沿用 SGD 的效率保证；a=0 标记为 `dig_zero_predictor`。
Geo3K 使用答案代理，ScienceQA 有解答时使用解答代理，参考答案和解答不进入输入问题。
同卡两任务的时间不能当成独占 GPU 的算法加速比，正式效率结论仍需独占复测。
