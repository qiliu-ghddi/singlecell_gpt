# 安装

对于GHDDI成员, cluster上已经安装好了运行所用的conda环境, 详情见[contributing](contributing/index.md).

对于其他用户, 除了本文档, 也可以参照[scGPT](https://github.com/bowang-lab/scGPT)创建所需的环境.

## 安装scGPT运行环境

预训练单细胞大模型例如scGPT, 要求 Python >= 3.7, R >=3.6.1. 如果要从头配置环境, 可以使用conda和pip.

1. 安装 [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. 创建一个新的conda环境, 以名称"flash-attn"为例:

   ```shell
   conda create -n "flash-attn" python=3.7.13
   ```
3. 激活环境:

   ```shell
   conda activate flash-attn
   ```
4. 安装 [PyTorch](https://pytorch.org/), [flash-attention](https://github.com/Dao-AILab/flash-attention)

   ```shell
   pip install torch==1.13.0
   pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
   # pip uninstall scgpt
   ```
5. 注意, flash-attention的安装对平台有要求, 建议用的机器RAM不少于96GB, CUDA驱动11.7. 另外假如CPU很多，在安装flash-attn可能会卡死:

   > If your machine has less than 96GB of RAM and lots of CPU cores, ninja might run too many parallel compilation jobs that could exhaust the amount of RAM. To limit the number of parallel compilation jobs, you can set the environment variable MAX_JOBS:
   > MAX_JOBS=4 pip install flash-attn --no-build-isolation
   >

   假如还是失败, 可尝试采用源码安装的方式来安装 `flash-attn`:

   ```shell
   git clone --recursive -b v1.0.1 git@github.com:Dao-AILab/flash-attention.git
   cd flash-attention
   python setup.py install
   ```
6. 卸载 `scgpt` 包. 对于需要修改 `scgpt` 源码包的任务, 我们采用本地源码导入, 因此需要uninstall之前pip所安装的scgpt模块.

   ```
   pip uninstall scgpt
   ```
7. 安装其他依赖

   ```
   pip install wandb
   ```
