# 单细胞基因表达数据集的下载和预处理——以CELLXGENE数据为例

[TOC]

按照scgpt论文和代码中的方法来下载cellxgene数据集。最初的来源为scgpt的[预训练分支](https://github.com/bowang-lab/scGPT/tree/dev-temp/data/cellxgene)。虽然当前的scGPT的[主分支](https://github.com/bowang-lab/scGPT/tree/main/data/cellxgene)也有该部分相应的脚本，和前面的内容稍有不同，我们的预训练分支是以前面分支的为基准进行改造而来的。

我们自己的数据和下面介绍的scgpt所下载的CELLxGENE数据在下载方式、文件组织结构、版本、规模、来源等均有不同，下面的内容仅供参考，具体基于我们自己数据的获取和预处理，将在下一个教程中来说明。



# 总流程

根据原Repo中的说明和代码，单细胞基因表达数据集的下载和预处理总流程如下：

1. 第一个步骤是按照细胞类型来queyr细胞索引文件 | Query index，运行`./build_soma_idx.sh`, `array_download.sh ` 
2. 分批（分块）下载数据集 | Download the Dataset in Chunks
3. 将 `AnnData` 转换为 `scb`，即`.parquet`类型的文件，便于后续的高效处理 | Build the `scb` Files。



# 结果

第一步，是按照细胞类型来queyr细胞索引文件。query idx的输入为`query_list.txt`文件，里面定义了需要query的什么类型的关键词，例如heart，blood，brain，lung，kidney，运行 `./build_soma_idx.sh` 输出的结果为下列文件：
```
blood.idx
brain.idx
heart.idx
intestine.idx
kidney.idx
lung.idx
others.idx
pan-cancer.idx
pancreas.idx
```
每个文件中都是query后的id，每一行都是一个8位数字的ID，例如`intestine.idx` 的第一行为51683484。注意，我们自己的数据是通过另外的途径下载下来的，这一步骤在处理我们自己数据时可以跳过。


第二步，执行下载脚本，下载后的结果`.h5ad`文件保存在`DATA_PATH="/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/data/cellxgene"`，文件列表详见[这里](notebooks/stat_downloaded_scgpt_human_ds.html)。

最后，Build之后的结果保存在`/home/qiliu02/ghddixcre_singlecell_gpt/data/cellxgene/scb_strict`，对于每个下载下来的`.h5ad`文件，我们生成了对应的`.parquet`格式的文件，便于后面用Hugging Face的`datasets`库进行处理。以及从中提取json格式的metadata，另外创建了一个`all_counts`目录，将所有的`.parquet`文件统一软连接到该目录下，例如"/home/qiliu02/ghddixcre_singlecell_gpt/data/cellxgene/scb_strict/blood/all_counts"。



# 步骤

原文中总的步骤如下：

```bash
cd /scratch/ssd004/datasets/cellxgene/
source env/bin/activate
INDEX_PATH="/scratch/ssd004/datasets/cellxgene/index"
DATA_PATH="/scratch/ssd004/datasets/cellxgene/anndata"
QUERY_PATH="query_list.txt"
./build_soma_idx.sh $INDEX_PATH $QUERY_PATH
sbatch array_download.sh $INDEX_PATH $DATA_PATH $QUERY_PATH

sbatch array_build_scb_filtering.sh
```



## Query index

在正式的处理之前需要安装和配置好本地环境，这里假定我们工作路径为：

```
cd /home/qiliu02/ghddixcre_singlecell_gpt/prepare_cellxgene_data
```

激活Conda环境：

```
conda activate /home/cliang02/work/software/common/proglang/mambaforge/envs/cre
# or
conda activate /home/qiliu02/miniconda3/envs/flash-attn
```


1. 首先是query index

```{bash}
INDEX_PATH="/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/data/scgpt_cellxgene/index"  # 要保存的路径
QUERY_PATH="data/query_list.txt"
./build_soma_idx.sh $INDEX_PATH $QUERY_PATH
```



## Download the Dataset in Chunks


2. 其次是Download the Dataset in Chunks

执行脚本：

```
./array_download_partition.sh
```

- 我们以分块的方式下载数据集；每个分块最多包含200000个单元，可以通过修改 `download_partition.sh` 文件 （被`array_download_partition.sh`调用）中的 `MAX_PARTITION_SIZE` 来调整分块大小。
- 在运行脚本之前，需要修改 `array_download_partition.sh` 文件中的 `DATA_PATH`、`QUERY_PATH` 和 `INDEX_PATH`。
- 确保 `INDEX_PATH` 和 `QUERY_PATH` 与上一步保持一致。
- `DATA_PATH` 是用于存储下载数据集的目录路径。下载的数据集将以 `h5ad` 格式存储。





## Build the `scb` Files

最后是Build the `scb` Files，

- 我们预处理数据集，然后将 `h5ad` 转换为 `scb`（用于高性能I/O的单细胞库）。
- 在运行脚本之前，需要修改 `array_build_scb.sh` 文件中的 `DATA_PATH`、`OUTPUT_PATH`、`QUERY_PATH` 和 `VOCAB_PATH`。
- 确保 `DATA_PATH` 和 `QUERY_PATH` 与上一步保持一致。
- `OUTPUT_PATH` 是存储 `scb` 文件的路径。
- `VOCAB_PATH` 是词汇表文件的路径，用于将基因ID映射到标记ID。



原文里使用的是作业提交系统，提交作业到集群：

```{bash}
sbatch array_build_scb.sh
```

我们没有使用sbatch作业提交系统，而是运行:

```
bash array_build_scb.sh
```



# 原文翻译

## Cellxgene 数据集

- Cellxgene 普查是一套公开可用的单细胞 RNA-seq 数据集，来源多样，涵盖超过 5000 万个细胞，取自各种组织和供体。
- 为了进行模型训练和验证，我们使用了日期为 `2023-05-08` 的 Cellxgene 普查版本。我们选择该数据集的标准是人类细胞，并基于报告的总体组织类型或疾病类型。
- 为了构建我们特定组织的基础模型，我们仅选择了来自心脏、肾脏、肺、胰腺、血液、大脑和肠道等七种不同组织的健康细胞。总共覆盖了 2280 万个细胞，不同组织的细胞数量从 21 万到 1320 万不等。整个人体模型是利用所有 3510 万个无疾病的人类细胞构建的。为了训练泛癌症基础模型，我们筛选了带有癌症疾病类型的细胞。这个查询结果为训练数据集提供了 570 万个细胞，代表 22 种不同的癌症类型。

## 预训练配置

- 工作流程如下：

  1. 根据查询构建细胞索引文件
  2. 分批（分块）下载数据集
  3. 将 `AnnData` 转换为 `scb`
- `query_list.txt` 记录了从 Cellxgene 普查中检索细胞图谱的查询。
- `build_soma_idx.sh` 为普查收集的所有健康人类细胞构建索引。
- `download_partition.sh` 使用给定的索引文件分批（分块）下载数据集，默认每个文件的最大分区大小为 200000 个细胞。

  - 设计为在作业数组模式下运行，每个作业下载一个查询。
- 我们扩展了 Cellxgene 普查的词汇：

  1. 继承原始 Cellxgene 词汇的顺序
  2. 添加从 Cellxgene 普查中新引入的基因
  3. 生成新的 JSON 格式的词汇文件
- 具体过程可以在 `dev_notebook/enrich_vocab.iypnb` 中找到
- 泛癌症集合包括 22 种癌症类型：支持的癌症类型可以在 `cancer_list.txt` 中找到