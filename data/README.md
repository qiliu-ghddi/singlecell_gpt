1. 将要准备的数据（.h5ad）放在raw中, 如示例中`data/raw/8`这样


```python
cd data
conda activate <env>  # 参考教程里的激活conda env，激活运行我们的scgpt环境
python build_large_scale_data.py --input-dir "raw/" --output-dir "./preprocessed" 
python binning_mask_allcounts.py --data_source "./preprocessed/all_counts/"

```