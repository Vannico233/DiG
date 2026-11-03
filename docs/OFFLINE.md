# CPU 联网准备，GPU 离线运行（共享目录）

适用于当前服务器：CPU 区是 Linux x86_64 / Python 3.9.19；GPU 区有两张 H200，不能联网；两区能看到同一个持久目录。

做法是在共享仓库中安装一整套 Python 3.11 环境。CPU 区负责安装、下载和整理数据，GPU 区直接使用已经安装好的环境。系统的 `/etc/inspire` 和 Python 3.9 保持原样；此流程不需要 sudo。

例如仓库位于 `wangningchong/DiG`，新环境就是该目录下的 `.runtime/py311`，权重和数据在 `assets`。
安装包缓存、HF 缓存、构建临时目录、Torch/Triton/CUDA 缓存也明确指向这个仓库下的路径；继承的 pip 安装目标和用户配置不会把依赖转装到公共环境。
准备脚本在子进程中设置这些路径，不修改系统 Python、别人的环境或 shell 启动文件；GPU 激活只影响当前终端及其子进程。

这属于安装目录和依赖隔离，不是文件系统访问隔离：不能承诺所有第三方程序绝对不在目录外写任何元数据。
例如 micromamba 可能在用户目录的 `.conda/environments.txt` 登记一条环境路径；这不代表把环境安装在用户目录。
如果需要严格禁止目录外的一切写入，需要服务器提供容器或访问权限限制。共享目录中其他人是否能读写你的文件，仍由服务器账号和权限决定。

**前提：两区必须用相同的绝对路径访问仓库，且都是兼容的 Linux x86_64 环境。**
在两区分别运行 `pwd -P` 检查路径。准备后不要移动或重命名仓库；Python 环境和项目安装信息含有绝对路径。
如果同一个共享目录在两区挂载路径不同，请先提供两边路径，再安排环境迁移，不能直接复制一个环境后假定可用。

## 1. 在 CPU 联网区执行

已经在 `DiG` 仓库内时：

```bash
git pull --ff-only
bash scripts/prepare_cpu_online.sh
```

默认 Python 3.9 可以启动这个脚本，不用先执行 `source .venv/bin/activate`，也不用先运行旧的 `bootstrap_h200.sh`。

脚本会完成以下工作：

| 内容 | 放在哪里 | 用途 |
| --- | --- | --- |
| Python 3.11、pip、C/C++ 编译器 | `.runtime/py311/` | 两区共同使用的独立环境；编译器供 Triton 在 GPU 上首次运行时使用 |
| CUDA 12.8 版 PyTorch 2.10.0、torchvision、HF 库及依赖 | 同一环境内 | CPU 区可以安装 GPU 版软件；此处不要求检测到 GPU |
| FLA 0.6.0、Triton、causal-conv1d 1.7.0 | 同一环境内 | causal-conv1d 使用官方匹配 wheel，检查 SHA256，无需在 CPU 区编译 CUDA 扩展 |
| Qwen3.5-4B 权重、tokenizer、processor | `assets/model/` | GPU 区从本地加载 |
| GSM8K 与 MATH numeric | `assets/data/` | 下载后完成清洗、长度过滤和 train/calibration/audit/validation/test 分割 |
| HF 下载缓存和固定 revision | `assets/hf-cache/`、`assets/download_lock.json` | 下载中断后复用已有内容，保留实验来源 |
| 安装版本清单与准备完成记录 | `.runtime/` | 核对依赖、权重、数据、代码和配置是否变化 |

数据准备只加载 tokenizer/processor，不把 4B 模型放入 CPU 内存做推理，不需要 GPU。
网络需能访问 Hugging Face、PyPI、PyTorch 下载站、GitHub 和 conda-forge；“能访问 HF”不等于这些站点都可访问。

最后必须看到：

```text
CPU_READY: all shared files are prepared.
```

并存在 `.runtime/CPU_READY.json`。**下载报错或脚本中途退出，不算准备完成。**
可在 CPU 区重新运行同一条命令；HF 会复用锁定版本和缓存，已安装的软件包通常不会重新下载。
如果刚中断的是预编译 wheel 下载，该文件会重新下载并核对摘要。
不要在 GPU 训练同时修改这个共享环境或重新执行准备脚本。

准备完成记录表示文件和环境已就绪，**不表示 H200 驱动和算子已经通过运行验证**。

## 2. 进入 GPU 离线区

进入同一个共享 `DiG` 目录，每开一个新终端都执行：

```bash
source scripts/activate_offline.sh
python --version
python scripts/offline_setup.py gpu
```

Python 应显示 3.11。激活脚本同时设置 HF 离线模式，禁止 pip 从网上安装，关闭 Hub 算子下载，使用 CPU 区已经安装的本地快速算子。
它适用于 bash 和 zsh，不修改 shell 的启动配置。原有 conda 提示符可能仍显示 `/etc/inspire`，以 `which python` 和 `python --version` 为准；退出该终端即可结束这次 PATH 切换。

检查脚本会核对路径、版本、权重和数据摘要，并在两张 GPU 上做 BF16 运算，检查快速算子是否成功导入。
驱动是否支持选定的 CUDA 运行时仍需以这里的实际结果为准。H200 型号本身不能证明驱动兼容。
该方案不要求另装完整 CUDA toolkit 或运行 `nvcc`；服务器仍须提供有效的 NVIDIA 驱动及其运行库。

通过后再执行模型级检查：

```bash
python -m dig_rl.launch --stage preflight --output runs/pilot01
python -m dig_rl.launch --stage smoke --output runs/pilot01
```

preflight 才会真正加载 Qwen、运行生成与反传，验证快速算子执行路径。
FLA/Triton 在第一次 GPU 运行时会本地编译、生成缓存，因此首次启动较慢；这不需要联网。
不要省略预检，也不要把“文件准备好”当成“已经测过模型训练”。

## 3. 审计、训练、评估

```bash
python -m dig_rl.launch --stage audit --output runs/pilot01
```

先查看 `runs/pilot01/audit/gsm8k/decision.json` 和 `runs/pilot01/audit/math_numeric/decision.json`。随后：

```bash
python -m dig_rl.launch --stage train --output runs/pilot01
python -m dig_rl.launch --stage evaluate --output runs/pilot01
python -m dig_rl.summarize --run runs/pilot01
```

全部使用本地权重和整理好的 JSONL 数据。GPU 区不用执行 `git pull`、安装脚本或 `dig_rl.prepare`。
审计门槛、预算、结果查看和恢复命令继续使用 [实验运行指南第 3 节起](SERVER.md#3-预检和-32-题流程检查)。

## 常见问题

- **CPU 区没有 GPU / 没有 nvcc：**正常；新脚本不依赖这些条件。不要因此改装 CPU 版 PyTorch。
- **GPU 区 Python 还是 3.9：**在仓库根目录重新 source；检查 `which python` 是否指向 `.runtime/py311/bin/python`。如果设置过 `python` 的 shell alias，先去掉该 alias。
- **缺少 CPU_READY：**回 CPU 区看准备脚本最先出现的错误并重跑，不能手动创建这个文件跳过检查。
- **网络下载失败：**保存具体报错和站点；不要关闭 TLS 验证。离线区无法补齐缺失的包。
- **CUDA 不可用或扩展导入失败：**保存 `nvidia-smi` 和检查脚本完整输出。需要根据驱动或库的实际报错处理；不要直接开始长实验。
- **路径/依赖/代码/配置摘要不匹配：**回 CPU 区核对改动后重新准备。实验开始后保持环境固定；新增实验配置应独立规划，不与已有 run 混用。

依据：[micromamba 官方安装说明](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)、[PyTorch 官方版本组合](https://pytorch.org/get-started/previous-versions/)、[causal-conv1d v1.7.0 官方预编译包](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)、[HF 离线环境变量](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。
