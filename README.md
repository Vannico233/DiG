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
