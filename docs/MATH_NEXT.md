# 数学机制验证：在线分块预测与零预测对照

2026-11-06 的 `round2_20261106_memfix` 已完成。64 题验证结果：
GSM8K base/RLOO/DiG = 59/58/58；MATH = 56/57/55。不是官方测试成绩。
四个训练任务的生成耗时占约 96.4%–98.5%；MATH 只有 2/3 次更新。
参考解答梯度乘全局非负系数，在拟合样本上的平方误差相对降低仅约
GSM8K 0.000233%、MATH 0.0000125%。这是未中心化的拟合指标，
不是 held-out 预测收益或方差降低证书，也不证明所有梯度代理都无效。

## 对照与协议

| arm | rho | 预测项 | 前置校准 |
|---|---|---|---|
| rloo | 1 | 无 | 无 |
| zero | 0.5 | 恒为零 | 无 |
| online | 0.5 | 每个 decoder layer 的历史梯度预测尺度 | 无 |

从相同初始化开始，三个 arm 的问题与回答随机流使用同一约定；抽样决定发生在当前回答生成之前。
保留 G=8、AdamW 1e-6、LoRA r16/alpha32、固定 loss_normalizer=2048、temperature=1 原始 softmax、
不启用 thinking；GSM8K 输出上限 4096，MATH 8192。
每步候选题 B 从 32 改为 **8**，缩短两次更新及保存之间的间隔；三个 arm 一致。
不能把这一轮当作与上一轮同配置的消融，也不声称 B=8 是顶会标准最优配置。
默认 **16 次更新**，每个 arm 最多 4 小时训练保护预算。预算在更新之间检查，可能超出一批时间；
未达到 16 步，pipeline 明确返回失败并保存部分结果，不能称作完成匹配步数比较。

验证使用训练集预先留出的 256 题、每题一次 greedy；取最终 checkpoint，不按验证成绩挑选 checkpoint。
官方测试为 GSM8K 的完整 test 和 MATH-500，通过独立 `test` 命令对 base 和全部三个 arm 执行。
不根据 test 选择模型或参数，不在这轮内调参后重测 test。

## 预测器

使用之前已完成更新的选中问题，按 decoder layer 维护 uu/ug 的 EMA，衰减 0.9；
拟合非负系数，加入跨块平均 uu 的 1% ridge，系数上限固定为 1。
本次更新开始时冻结 A_t，随后才抽取 Bernoulli z、生成回答并计算梯度：

```text
h_i = A_t u_i + z_i/rho (g_i - A_t u_i)
E_z[h_i | history, question, potential rollout] = g_i
```

一次完整更新只做一次 AdamW step。当前批次的 uu/ug 仅供下一次更新使用；
不能在当前问题或批次上拟合后立即用于同一批次。成本记录在训练时间中。
第一步 A=0，online 与 zero 的采样和参数更新应相同，测试覆盖此项。
恒等式只保证梯度估计的条件均值；不保证 AdamW 更新、准确率或效率。
逐步记录冻结预测器相对于新 RL 梯度的残差平方和；这是针对单个带噪梯度的
out-of-fit 检查，不是潜在真梯度误差、独立审计或统计证书。
不按当前奖励过滤全对/全错组，也不反复补采直到出现混合组，以免改变抽样分布。

## 生成与资源

原循环每个 token 都向 GPU 传随机数并在 CPU 等待有限值检查。
新采样器预先传输相同的逐回答随机流，在每批结束后检查异常；
随机分布及随机数顺序不变。异常批次报错，不能用于评分或训练。
没有实现连续补入请求，也没有保证 GPU 加速比例。
HF 连续批处理及 vLLM 需要另外验证 Qwen3.5 混合注意力、LoRA 和权重同步的一致性；
这一轮不安装新的推理引擎或切换模型。

分别实测 legacy batch16、buffered batch16/32/64：相同 8 道 calibration 题，每题 8 条回答，
使用完整 4096/8192 上限，然后执行最长 calibration 输入的完整反向检查。
只在通过、峰值 reserved <60 GiB 的候选中按有效 tokens/s 选最快者；同数据集三个 arm 使用相同选项。
记录 useful_decode_fraction = 实际回答 tokens / 固定批次分配的 decode slots，量化 EOS 后的空算。
一次测速不是最坏输入显存保证；OOM 候选被拒绝，其他异常停止实验。

每个 worker 独占一张 GPU，四卡动态排队，不干扰已有 GPU 进程。
队列包含 8 个 profile、6 个训练和 2 个初始模型验证任务；尾部可能不足四个任务。
profile/eval 及四卡阶段成本单独记录，不能隐去后声称端到端加速。
每次更新保存参数、优化器、预测器、采样及 PyTorch RNG；同版本可恢复。
代码/依赖/数据/协议变化必须新建 run，旧实验保留。

## CPU 同步（不重新下载）

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG &&
git fetch origin main &&
git merge --no-edit origin/main &&
source scripts/activate_offline.sh &&
python scripts/offline_setup.py seal
```

## GPU 启动与监测

```bash
cd /inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG
mkdir -p .runtime
nohup bash scripts/start_math_next.sh runs/math_online_20261106 \
  >> .runtime/math_online_20261106.log 2>&1 < /dev/null &
echo "Launcher PID=$!"
```

```bash
source scripts/activate_offline.sh
python scripts/math_next.py status --output runs/math_online_20261106
tail -n 30 .runtime/math_online_20261106.log
tail -n 20 .runtime/math-next-preflight.log
nvidia-smi
```

初步结果：

```bash
python scripts/math_next.py summarize --output runs/math_online_20261106
```

所有 arm 达到固定步数后，按冻结协议运行完整官方测试（不再训练）：

```bash
nohup python -u scripts/math_next.py test --output runs/math_online_20261106 \
  >> .runtime/math_online_test.log 2>&1 < /dev/null &
```

`results.json` 是 validation，`test_results.json` 才是 official test。
比较 online−rloo 和 online−zero 的题目配对区间、实际 tokens、训练时间及滞后预测残差。
这是新方法的单种子机制验证，尚无 GPU 结果，不是 SOTA 或一般效率结论。
主实验仍需固定协议、多种子与更长训练。
