# 四卡探索性对照

`scripts/run_exploratory.py` 用于审计未通过时仍观察训练效果的独立实验。
原来的 `dig_rl.launch` 门槛策略不变；原始 `decision.json` 不修改。
审计测量冻结初始模型上的梯度关系和成本，不能替代训练后准确率。

## 固定协议

- GPU 0：GSM8K RLOO；GPU 1：GSM8K DiG。
- GPU 2：MATH numeric RLOO；GPU 3：MATH numeric DiG。
- DiG 使用原校准尺度 `a`，统一固定 `rho=0.5`。这是观察审计结果后设定的探索性配置，不是通过门槛的最优配置。
- 如果 `a=0`，结果标记为 `dig_zero_predictor_exploratory`：此时是随机抽样并校正权重的 RLOO，不能归因于示范预测收益。
- 四组分别最多训练 4 小时或 200 步；每 5 步以及第 1 步保存断点。整批计算可能超过预算边界，会记录超额。
- 保持原始 SGD、学习率调度、LoRA、生成参数、训练数据及随机种子不变。
- 同数据集的两臂核对初始参数摘要。直接调用现有 `dig_rl.train.update`，不复制或修改梯度估计公式。
- 每组在固定 256 道 validation 题上评估，每题采样 4 条回答，计算平均成功率（随机 pass@1，而非 pass@4）。RLOO 工作进程另评估同一初始模型，供两臂比较。test 不访问。
- 只评估最终 checkpoint，不按 validation 成绩选择 checkpoint。单种子、探索性结果不等同于多种子确认性结论。

两个成本账同时保留：`training_seconds` 是训练阶段墙钟耗时；`end_to_end_seconds` 对 DiG 加上源审计的全部耗时。
模型加载及独立评估不计入这两个训练成本账；评估的生成耗时单独记录。
等训练时间实验不能自动写成等端到端成本加速。

## 启动

先在 CPU 空间同步代码并重新执行 `python scripts/offline_setup.py seal`（使用激活后的共享环境）。
本次只增加脚本和测试，`src/` 核心代码未改变。新入口仍严格核对原审计的模型、配置、数据、库版本和核心代码摘要，
还重算校准 `a`。新入口自身摘要、运行参数和原始审计文件摘要另存进实验清单。

在 GPU 空间进入 DiG 并激活，然后启动新目录：

```bash
source scripts/activate_offline.sh
python scripts/offline_setup.py gpu
nohup python -u scripts/run_exploratory.py launch \
  --audit-run runs/pilot_20261104_062546 \
  --output runs/explore_20261104_rho05 \
  --gpus 0 1 2 3 --rho 0.5 \
  --train-seconds 14400 --max-steps 200 --save-every 5 \
  --eval-problems 256 --eval-samples 4 \
  > .runtime/explore_20261104_rho05.log 2>&1 < /dev/null &
```

该命令自动执行四组训练、训练后评估、两个初始模型评估和汇总。已有目录会被拒绝，不覆盖旧实验。
GPU 开始训练前应确保 0、1、2、3 没有其他任务。

## 监测和恢复

另一个终端进入同一 DiG 目录：

```bash
python scripts/run_exploratory.py status --output runs/explore_20261104_rho05 --watch
```

监测不导入模型，每 10 秒刷新；Ctrl+C 只退出监测。每组 `.log` 有每步指标和逐题评估进度。
`progress_*.json` 包含阶段、进度、更新时间和 PID；总状态在 `status.json`。

只有任务已停止时才恢复；启动器及每个工作进程都有独占锁，防止同目录重复训练：

```bash
source scripts/activate_offline.sh
nohup python -u scripts/run_exploratory.py resume \
  --output runs/explore_20261104_rho05 \
  >> .runtime/explore_20261104_rho05.log 2>&1 < /dev/null &
```

恢复沿用原清单，禁止更换代码、配置和预算。训练从最近断点恢复，评估从已完成题目继续。
因恢复而退回的训练日志尾部会另存备份。

### 用空闲四卡补齐中断的初始模型评估

如果所有训练和训练后评估已完成，只剩一个数据集的初始模型评估，可使用
`scripts/finish_base_eval.py`。先在 CPU 空间更新代码，激活共享环境并重新运行
`python scripts/offline_setup.py seal`。GPU 实例恢复后确认四卡没有其他任务，执行：

```bash
source scripts/activate_offline.sh
python scripts/offline_setup.py gpu
nohup python -u scripts/finish_base_eval.py \
  --output runs/explore_20261104_rho05 --dataset math_numeric --gpus 0 1 2 3 \
  >> .runtime/finish_base_eval.log 2>&1 < /dev/null &
```

这个独立入口保留原 `run_exploratory.py` 及训练 checkpoint；每题仍使用原来的
全局题号对应种子和四条回答的批次。已有完整题目不再生成，剩余题目分给四卡。
每完成一题就原子保存独立文件；被中断后重复同一命令，复用已完成的题目。
启动器持有原实验锁，防止与原 `resume` 同时运行；子进程继承锁，避免父进程被强制结束后重复启动。
每个进程检查库、核心代码、数据及初始参数身份。所有分片完成后，按原题序合并，保留原前缀备份，
调用原评估汇总和比较逻辑，生成完整 `results.csv`、`comparisons.json`。
改变并行调度不承诺不同硬件运行时的逐位确定性；该入口不改变采样分布或评估口径。

监测这个补齐任务（原监测器在合并前仍显示原来的前缀数量）：

```bash
tail -n 20 .runtime/finish_base_eval.log
cat runs/explore_20261104_rho05/eval/math_numeric_base/parallel/status.json
```

分卡日志在同一 `parallel/slot-*.log`，每题结果在 `parallel/parts/`。
`parallel/status.json` 中的本次墙钟耗时与原评估 `generation_seconds` 分开记录；
后者是各题生成用时之和，多个 GPU 并行时不能直接视为实际等待时长。
本入口只补齐初始模型评估，不重新训练，也不启动新实验。

## 结果

- `experiment.json`：固定配置、四卡分工、原始失败门槛、脚本与审计文件摘要。
- `train/*/metrics.jsonl`：训练曲线；`summary.json`：步数、参数变化、两种成本账。
- `eval/*/summary.json`：初始模型和训练后模型的验证成绩；`predictions.jsonl`：逐题输出。
- `results.csv`：四组成绩、相对初始模型变化、耗时、token、有效方法和 `a/rho`。
- `comparisons.json`：同题配对的 DiG−RLOO、RLOO−初始、DiG−初始差异及题目 bootstrap 区间。

所有字段均报告真实执行值；不把被拒绝的门槛改为通过，也不隐藏零预测器或审计成本。
