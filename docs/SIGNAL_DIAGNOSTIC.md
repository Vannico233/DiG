# 短程训练信号诊断

现有四小时实验的 GSM8K RLOO 有 21/50 次更新梯度为零，MATH RLOO 有 6/23 次。
初始与训练后验证成绩接近。`diagnose_signal.py` 用短程对照检查更新尺度和实际耗时，
不把小验证集结果当作正式成绩，不重用失败审计来宣称 DiG 优势。

## 固定设计

| GPU | 数据集 | 初始学习率 |
|---|---|---|
| 0 | GSM8K | 0.01（原设置） |
| 1 | GSM8K | 1.0（100 倍尺度探针） |
| 2 | MATH numeric | 0.01 |
| 3 | MATH numeric | 1.0 |

四组均使用完整 RLOO、原始 SGD 和 `eta0/sqrt(step+1)` 调度。核心 `src/` 不变；
数据、LoRA 初始化、生成分布、题目抽样与回答随机种子流保持一致，只有初始学习率不同。
较大学习率只是诊断用的预先固定对照，不是正式训练推荐。不同更新后策略会生成不同回答。

每组最多 4 次更新或 3600 秒训练，先到者停止；每次更新保存 checkpoint。
训练后只评估固定 validation 前 32 题，每题四条回答；原模型在这些题目的输出从上一轮直接复用。
启动前校验参考模型、初始化、数据、库、原评估种子和回答数，不额外评估初始模型，不访问 test。
单种子、32 题的结果仅用于诊断；后续配置选择需要独立确认，不能作为正式性能提升结论。

每个训练候选组记录生成和反向耗时、回答奖励、是否奖励全部相同、token 数、截断、可解析情况、
峰值已分配显存。奖励全部相同会使该组 RLOO advantage 为零；组内不一致也不保证最终整步梯度非零。
原训练更新继续记录整步梯度范数、学习率、参数变化。主进程每约 15 秒记录一次 GPU 利用率、显存、功耗和任务阶段。

## CPU 同步和准备

先从 GitHub 获取并合并更新（保留本地报告提交），然后在共享仓库执行：

```bash
source scripts/activate_offline.sh
python -m pytest -q tests/test_signal_diagnostic.py tests/test_exploratory.py tests/test_finish_base_eval.py
python scripts/offline_setup.py seal
python scripts/diagnose_signal.py prepare \
  --source-run runs/explore_20261104_rho05 \
  --output runs/signal_lr_20261105
```

准备可在当前接续评估期间完成：只复制已经存在的前 32 题参考输出并验证来源，不改源实验。
已有输出目录会被拒绝；已成功准备后不要重复 prepare。

## GPU 执行

等待上一轮四卡接续评估全部完成，再在 GPU 实例执行：

```bash
source scripts/activate_offline.sh
if python scripts/offline_setup.py gpu; then
  nohup python -u scripts/diagnose_signal.py run \
    --output runs/signal_lr_20261105 \
    >> .runtime/signal_lr_20261105.log 2>&1 < /dev/null &
fi
```

脚本检查源实验完成状态并持有源实验锁；发现 GPU 上已有计算进程会退出，不并发抢占旧任务。
它不会自动停止任何进程，也不会排队到未来自动启动。中断后使用相同 `run` 命令恢复；
保持代码、清单和输入不变，训练从最近整步 checkpoint 接续，评估从已完成题目继续。
诊断日志中未提交整步的分组记录会在重算时覆盖，汇总只使用完成的更新。

```bash
python scripts/diagnose_signal.py status --output runs/signal_lr_20261105
tail -n 20 .runtime/signal_lr_20261105.log
```

每组日志位于运行目录下的 `gsm8k_lr0.log` 等，包含每组生成/反向耗时；
`profile/<job>/` 是逐组记录，`gpu_samples.jsonl` 是带时间和阶段的设备采样。
全部完成后自动生成 `signal_report.json`：两档学习率的参数变化、零梯度次数、各阶段耗时、
吞吐、验证成绩以及与初始模型和另一档学习率的配对差异。

这不是显存压力测试，也未改变生成批次或接入新的推理引擎。先用实测数据定位瓶颈，再决定是否
增加每次推理的题目数量或采用其他生成实现；9 GB 显存占用本身不能推导可叠加的任务数。
现有 Engine 只输入文本；VLM 实验还需要独立验证图像输入、数据和奖励协议。
