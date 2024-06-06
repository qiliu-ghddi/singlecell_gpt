import numpy as np
import pandas as pd

# ! ls /home/lushi02/project/sl_data/cellgene20230606/
meta = pd.read_csv("/home/lushi02/project/sl_data/cellgene20230606/meta.csv")
# print(meta)

# all human ：`Organism` 为 `Homo sapiens`, `Disease`为`normal`
query_otd1 = [
    "Organism == 'Homo sapiens'",
    "and",
    "Disease == 'normal'"
]
query_str_otd1 = " ".join(query_otd1)
print(query_str_otd1)

meta_quered_otd1 = meta.query(query_str_otd1)
# print(f"len: {len(meta_quered_otd1)}")

meta_quered_otd1['Cell Count'].sum()
print(meta_quered_otd1['id'].tolist())
print(f"len: {len(meta_quered_otd1)}")