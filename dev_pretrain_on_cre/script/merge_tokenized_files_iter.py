import os
import gc
import time
import logging
from pathlib import Path
import torch
from datetime import datetime

t_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(f"save/cre329_tokenized_merged/{t_stamp}")
save_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=f'{save_dir}/run.log',  # Specify the file name for the log
    level=logging.DEBUG,  # Set the minimum log level (you can adjust this)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import psutil

def get_memory_usage():
    memory = psutil.virtual_memory()
    memory_info = {
        "total": memory.total / (1024 ** 3),
        "available": memory.available / (1024 ** 3),
        "used": memory.used / (1024 ** 3),
        "free": memory.free / (1024 ** 3),
        "percent": memory.percent
    }

    return memory_info

def get_file_size(filepath):
    size_bytes = os.path.getsize(filepath)
    size_gb = size_bytes / (1024 ** 3)
    size_mb = size_bytes / (1024 ** 2)
    size_kb = size_bytes / 1024
    return size_gb, size_mb, size_kb


data_dir = Path("/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/dev_pretrain_on_cre_all_in_one/script/save/cre329_tokenized/20230731_235828")
files = data_dir.glob("*.pt")

params = [
    # {'id': 8, 'fn': f"{data_dir}/cre8_tokenized_data.pt"},
    # {'id': 385, 'fn': f"{data_dir}/cre385_tokenized_data.pt"},
    # {'id': 386, 'fn': f"{data_dir}/cre386_tokenized_data.pt"},
    
    {'id': int(str(f.stem).split("_")[0][3:]), 'fn': str(f)} for f in files
]

# # load and merge
# ds = {
#     'ids': [],
#     'genes': [],
#     'values': [],
# }

ids = [p['id'] for p in params[:]]
logging.info(f"ids of all datasets: {ids}")
logging.info(f"number of all datasets:{len(ids)}")
logging.info(f"="*50)

memory_info = get_memory_usage()
logging.info(f"... Before merging")
logging.info(f"Mem Total: {memory_info['total']} GB")
logging.info(f"Mem Available: {memory_info['available']} GB")
logging.info(f"Mem Used: {memory_info['used']} GB")
logging.info(f"Mem Free: {memory_info['free']} GB")
logging.info(f"Mem Percent: {memory_info['percent']} %")
logging.info(f"... Start merging")

t_global_s = time.time()

final_d = None
cnt = 0
for i, parm in enumerate(params[:]):
    try:
        t_start = time.time()
        
        id_ = parm['id']
        logging.info(f"... {i} ... id: {id_}")
        d = torch.load(parm['fn'])
        logging.info(f"file name: {parm['fn']}")
        logging.info(f"file size (GB, MB, KB): {get_file_size(parm['fn'])}")
        logging.info(f"d type: {type(d)}")
        logging.info(f"d.keys: {d.keys()}")
        logging.info(f"d['value'].shape: {d['values'].shape}")
        
        if final_d is None:
            final_d = d
        else:
            final_d = {
                "genes": torch.cat([final_d['genes'], d['genes']], dim=0),
                "values": torch.cat([final_d['values'], d['values']], dim=0)
            }
        cnt += 1
        t_end = time.time()
        memory_info = get_memory_usage()
        logging.info(f"After merging dataset {cnt} (id={id_}) Mem Percent: {memory_info['percent']} %")
        logging.info(f"Time elapsed: {t_end-t_start} s")

        del d
        gc.collect()
    except Exception as e:
        logging.error(f"Error! {e}")
        continue

logging.info(f"="*50)
logging.info(f"merged_tokenized_data type: {type(final_d)}")
logging.info(f"merged_tokenized_data shape: {final_d['values'].shape}")
torch.save(final_d, f"{save_dir}/cre329_tokenized_merged_numds{cnt}.pt")

t_global_e = time.time()
logging.info(f"Total Time elapsed: {t_global_e-t_global_s} s")
logging.info(f"Finished~")
