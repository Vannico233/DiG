# CPU 联网准备，GPU 离线运行（共享目录）

适用于当前服务器：CPU 区是 Linux x86_64 / Python 3.9.19；GPU 区已改为四张 H100 80GB，base Python 3.13.5，不能联网；两区能看到同一个持久目录。项目仍使用独立 Python 3.11，下载版本无需随 base Python 改变。本流程也适用于之前的两张 H200。

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
| PyTorch / CUDA wheel 及下载清单 | `.runtime/wheels/` | 大包先下载、续传、校验，再从本地安装；已完成包保留供重跑复用 |
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
可在 CPU 区重新运行同一条命令；锁定版本及已完成的缓存文件会复用，已安装的软件包通常不会重新下载。
第 3 步现在默认使用持久 curl 下载模型权重，详见下方“模型下载中断”。旧 Hub/Xet 流程的未完成模型分片不保证可以跨进程续传。
固定摘要的大包下载会保留 `.partial` 文件；重跑时，如果服务器支持 HTTP Range，就从保存位置续传。
所有固定摘要文件必须校验通过后才能安装；默认 urllib 方式遇到服务端忽略续传时，会提示并从头下载当前文件。可选 curl 方式遇到拒绝续传时则停止并保留进度。
不要在 GPU 训练同时修改这个共享环境或重新执行准备脚本。

准备完成记录表示文件和环境已就绪，**不表示 GPU 驱动和算子已经通过运行验证**。

### 下载很慢或想暂停更新

第 2 步仅固定的 PyTorch / CUDA 依赖就约 4.13 GB，之后还有模型权重；单个 cuDNN 包并不是全部下载量。
新版从 PyTorch 官方索引指向的 CDN 获取两个核心 wheel，从 PyPI 获取同版本 NVIDIA/Triton 依赖。
这避免所有依赖都经过 PyTorch 下载站，但不能保证你所在网络的速度一定提高。

如果当前仍在 `Downloading` / `Resuming`，可以按一次 Ctrl+C，等命令退出后再 `git pull --ff-only` 并重新执行准备脚本。
如果已经进入环境的 `Linking` 或 pip 的 `Installing collected packages`，先等该安装步骤完成。
保留 `.runtime` 和 `assets`；不要同时开第二个安装进程。

升级前的 pip 缓存中，已完整下载且摘要匹配的 wheel 会尝试复用；旧 pip 临时下载中未完成的部分不保证能接续。
安装包的持久续传针对固定摘要的 Torch/CUDA/causal-conv1d 文件；一般小型 Python 依赖仍由 pip 下载。模型权重使用独立的持久 curl 流程，小文件和数据集仍由 HF 管理。
已有完整的 `.runtime/wheels/torch-downloads.json` 时，新版会核对包版本、平台和清单完整性，直接复用清单，不再查询 PyPI 元数据。
首次查询的每个包信息会单独缓存，遇到短响应、断线或暂时性 HTTP 错误会重试。元数据查询失败不会删除或修改已保存的 wheel `.partial` 文件。
单独查看固定包下载计划（只读取元数据）可用：

```bash
.runtime/py311/bin/python scripts/torch_wheels.py --plan-only
```

如单独执行时仍需要项目证书，可在命令前加 `SSL_CERT_FILE="$PWD/.runtime/certs/ca-bundle.pem"`。

如果 curl 对实际下载站的测速较快，可以切换固定摘要文件的下载方式：

```bash
DIG_DOWNLOAD_TRANSPORT=curl bash scripts/prepare_cpu_online.sh
```

在原下载进程退出后执行。curl 直接续传同一个 `.partial` 文件，不需要重新下载已保存部分；它继承当前终端的代理变量，并使用项目证书。
完成后仍校验 SHA256 和已知文件大小。服务器拒绝续传时，curl 模式报错并保留进度，不自动从头覆盖。
这只切换固定摘要安装文件的下载，不改变 pip、PyPI 元数据查询或 Hugging Face 的下载方式；不保证一定提速。
curl 显示原生进度及传输结果；默认 urllib 模式同时显示最近报告区间的速度和本次连接平均速度，断线时输出具体原因。

如果经过代理下载 PyPI 的 NVIDIA/Triton 包仍然很慢，可以对这些包改用国内镜像直连：

```bash
DIG_PYPI_MIRROR=tuna DIG_DOWNLOAD_TRANSPORT=curl bash scripts/prepare_cpu_online.sh
```

`tuna` 使用清华镜像，也可选 `ustc`（中科大）、`aliyun`（阿里云）或 `huawei`（华为）。准备脚本只在自身及子进程中设置 pip 源，并只对这些镜像域名绕过代理；PyTorch、GitHub、Hugging Face 仍沿用已有代理设置。
固定 CUDA 下载清单保留官方 URL、版本、文件大小和 SHA256，只在取文件时替换镜像地址。已下载的同名 `.partial` 直接续传，最终必须通过原始 SHA256 校验；完整且校验通过的包不会重新下载。
镜像缺包、拒绝续传或校验失败时会报错，保留进度，不自动换包版本。若镜像不可用，重新以 `DIG_PYPI_MIRROR=official` 运行即可回到原来源。
这个选项也会让后续普通 pip 安装使用同一镜像，但不改变模型/数据的下载来源，不保证实际提速。
镜像说明：[清华 PyPI](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[中科大 PyPI](https://mirrors.ustc.edu.cn/help/pypi.html)。

出现 HTTP 403 时，同一个来源的重复重试通常无助于解决拒绝访问；已保存的 `.partial` 不会因这个错误被清空。
可让实际运行准备脚本的服务器先检测来源，再继续：

```bash
DIG_PYPI_MIRROR=auto DIG_DOWNLOAD_TRANSPORT=curl bash scripts/prepare_cpu_online.sh
```

它对第一个尚未完成的 PyPI wheel，从实际保存位置发起小范围请求，依次检查四个镜像直连和官方来源（继承已有代理）。每个来源最多短测 10 秒、正常响应最多取 2 MiB，不写入 wheel 文件；403、错误的 Content-Range、大小不符等结果会跳过。
随后选择本次短测中可用且速度最高的来源，继续原流程。若都不可用则停止并保留下载文件。首次准备尚无清单时，使用一个已知的小 NVTX 包探测。
短测只能反映当时的访问和速度，不保证整个文件持续同速，也不证明所有后续包都可访问；再次报错时可重跑自动检测。
单独检测而不安装可执行 `SSL_CERT_FILE="$PWD/.runtime/certs/ca-bundle.pem" .runtime/py311/bin/python scripts/pypi_routes.py`。
中科大在缓存未命中时可能重定向到清华，因此它们不一定是两条独立可用的下载路径；自动检测会跟随重定向检查最终响应。

### 模型下载中断（第 3 步）

默认 `DIG_MODEL_TRANSPORT=curl`：先从官方模型 API 读取已锁定 revision 下各个 `.safetensors` 的文件大小和 SHA256，保存到 `assets/model/.dig-downloads/`。每片权重使用 8 MiB 分段、最多 4 条连接下载。已有连续 `.partial` 前缀保留不动，之后的分段写入对应 `.chunks`，按顺序合并；重跑会复用二者。只有整个文件大小与官方 SHA256 都通过才会放入模型目录。

`DIG_MODEL_SOURCE=auto` 是默认来源选择：在执行脚本的服务器上并行测试官方来源（继承代理）、HF-Mirror 直连，以及已验证内容一致的魔搭直连/现有代理线路。每个探测最多 15 秒、2 MiB；只有最终 HTTP 206、Content-Range 和总大小符合请求的来源才参与选择。魔搭映射限定为本项目锁定的 Qwen3.5-4B revision 和两个已核对 SHA256 的权重，不推断其他版本相同。可显式指定 `official`、`hf-mirror`、`modelscope` 或 `modelscope-proxy`。镜像直连请求不会修改系统或终端的代理设置。

每段请求最多 90 秒，断流或超时后会保留已验证响应范围内的有效字节并切换下一个可用来源。重试上限统计**连续没有新字节**的请求（至少 12 次），有进度就重置；不会因为整个大文件累计断流四次而停止。忽略 Range、范围错误或错误页面不会追加到旧文件。每 10 秒报告保存量和正在接收的字节。可用 `DIG_MODEL_CONNECTIONS=1..8` 调整连接数；并行不保证突破平台或线路的总带宽限制。

```bash
git pull --ff-only
DIG_MODEL_TRANSPORT=curl DIG_PYPI_MIRROR=auto DIG_DOWNLOAD_TRANSPORT=curl bash scripts/prepare_cpu_online.sh
```

已完成的模型权重会核对 SHA256 后复用。配置、词表等小文件仍通过 Hub 下载；后续 Hub 快照明确排除权重，不会再次用 Xet 下载同一组大文件。数据集下载保持原流程。

该模式针对当前公开、非 gated 的 Qwen 模型，使用项目 CA，不关闭证书检查，不改变模型版本。候选来源均无有效进度或文件校验失败时停止并保留文件。准备脚本和模型下载均有进程锁，避免两个写入者互相覆盖；不要删除 `.runtime`、`assets` 或 `.dig-downloads`。离线准备记录不包含可变的下载缓存和锁文件。

如果已安装完第 2 步、在第 3 步失败，可直接后台恢复并自动继续第 4～6 步：

```bash
git pull --ff-only
bash scripts/resume_cpu_online.sh
tail -f .runtime/prepare-resume.log
```

恢复脚本检查现有 Python/PyTorch 环境，跳过已完成的软件包安装，启用自动模型选源和并行下载，使用 `nohup` 保持进程运行，日志按启动时间保留。`tail` 中 Ctrl+C 只停止查看日志，不停止后台安装；关闭网页终端通常不会中断 `nohup` 作业，但平台关闭实例或回收进程仍会停止。成功标准仍是日志末尾的 `CPU_READY` 和 `.runtime/CPU_READY.json`，仅显示“已启动”不代表配置完成。

前台恢复可用 `DIG_START_STEP=3 bash scripts/prepare_cpu_online.sh`。一般新环境仍执行完整准备脚本。HF 小文件和数据集默认禁用 Xet，使用普通 HTTP 下载；它们的网络重试次数可用 `DIG_NETWORK_ATTEMPTS` 配置，默认 12。

服务器访问下载源仍很慢时，可在网络较快的电脑下载相同权重，通过平台文件上传入口送到项目的 `.runtime/model-incoming/`。先在服务器核对文件大小和官方 SHA256，再放入 `assets/model/`；恢复准备会校验并复用完整权重。不要把尚未完成的上传文件当成模型，也不要用其他量化版或不同 revision 替代。先传一个小样本验证上传链路，避免把瓶颈从下载换成上传。

如果日志出现 `File reconstruction error: CAS Client Error ... error decoding response body`，说明当次仍走 Hub/Xet。旧下载器可能已清理未完成的临时文件，不能承诺找回日志里显示的所有下载字节；也不能把 Xet 重建的临时文件当作连续前缀直接接到 curl 文件上。新模式的持久续传从它自己保存的 `.partial` 开始。

`DIG_MODEL_TRANSPORT=hub` 可显式回到旧快照方式。仅设置 `DIG_DOWNLOAD_TRANSPORT=curl` 不控制模型下载；模型选项是独立的。新方式避免 Xet 重建错误并保留下载进度，但不保证代理到 Hugging Face 的线路一定提速。

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
驱动是否支持选定的 CUDA 运行时仍需以这里的实际结果为准。GPU 型号本身不能证明驱动兼容。
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
- **第 3 步提示 SOCKS proxy 缺少 `socksio`：**更新代码后重跑准备脚本。`server` 依赖已包含 `httpx[socks]`，会在第 2 步补齐该小包并复用已有大包。HTTPX 会读取 `HTTP_PROXY`、`HTTPS_PROXY` 和 `ALL_PROXY`；即使 HTTPS 使用 HTTP 代理，初始化时也可能因另一条 SOCKS 配置而要求该依赖。脚本不会因此删除或修改你的代理设置。[HTTPX SOCKS 文档](https://www.python-httpx.org/advanced/proxies/#socks)
- **第 3 步出现 `UNEXPECTED_EOF_WHILE_READING`：**这表示连接异常结束，需要与证书不可信区分。准备程序对临时连接错误最多尝试 4 次，每次重建 HF 客户端；单次仓库版本查询设 30 秒超时，每解析成功一个仓库就保存其固定 revision。模型和数据下载重试时复用 HF 缓存。认证失败、仓库不存在和证书校验失败不会通过这层重试跳过；TLS 校验始终开启。curl 成功只能证明那一次连接可用，不能保证 Python 后续连接稳定。
- **第一步出现 `CERTIFICATE_VERIFY_FAILED`：**更新代码后重跑准备脚本。它会合并当前 Python、常见 Linux 系统路径以及已安装 certifi 中的可信 CA，保存到 `.runtime/certs/ca-bundle.pem`，供下载器、pip、HF 和 micromamba 使用；不会修改公共环境或系统证书。此修复解决已安装证书未被当前 Python 使用的情况，不能补出服务器根本没有的证书。
- **CUDA 不可用或扩展导入失败：**保存 `nvidia-smi` 和检查脚本完整输出。需要根据驱动或库的实际报错处理；不要直接开始长实验。
- **路径/依赖/代码/配置摘要不匹配：**回 CPU 区核对改动后重新准备。实验开始后保持环境固定；新增实验配置应独立规划，不与已有 run 混用。

如果证书仍然失败，先只检查出错的下载连接（包含重定向，不下载完整安装包）：

```bash
python3 scripts/offline_setup.py tls-check
```

保留输出和 `.runtime/certs/sources.json`，其中记录证书来源，不记录代理密码或 token。
若服务器网络使用专门的 CA，需从服务器管理员处取得可信的 **CA 证书文件**，将其路径传入后重跑。
例如管理员已把文件提供在项目内的 `.runtime/certs/server-ca.pem` 时：

```bash
DIG_CA_BUNDLE="$PWD/.runtime/certs/server-ca.pem" bash scripts/prepare_cpu_online.sh
```

不要把网站返回的任意证书直接当作可信根，也不要使用关闭 SSL 校验的配置绕过此错误。
该项目只复制公开 CA 证书，不需要服务器私钥。

依据：[micromamba 官方安装说明](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)、[PyTorch 官方版本组合](https://pytorch.org/get-started/previous-versions/)、[causal-conv1d v1.7.0 官方预编译包](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)、[HF 离线环境变量](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。
证书配置依据：[Python SSL](https://docs.python.org/3.9/library/ssl.html)、[pip 证书配置](https://pip.pypa.io/en/stable/topics/https-certificates/)、[HTTPX 证书环境变量](https://www.python-httpx.org/environment_variables/)。
