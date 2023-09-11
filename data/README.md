# 说明

使用我们数据预处理脚本来将.h5ad数据处理成scGPT pretraining需要的input embedding数据集，步骤如下：
1. 将要准备的数据（.h5ad）放在raw中, 如示例中`data/raw/8`
2. `cd data`, 运行`python build_large_scale_data.py`
3. 然后运行`python binning_mask_allcounts.py`

例：
```shell
cd data
conda activate <env>  # 参考教程里的激活conda env，激活运行我们的scgpt环境
python build_large_scale_data.py --input-dir "raw/" --output-dir "./preprocessed" 
python binning_mask_allcounts.py --data_source "./preprocessed/all_counts/"
```

## Training

```
python pretrain_scGPT.py --data_source="../data/binned/" --valid_size_or_ratio=0.3

```