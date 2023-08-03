import os
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

# load and merge
ds = {
    'ids': [],
    'genes': [],
    'values': [],
}

for i, parm in enumerate(params[:10]):
    try:
        id_ = parm['id']
        logging.info(f"... {i} ... {id_}")
        d = torch.load(parm['fn'])
        logging.info(f"file name: {parm['fn']}")
        logging.info(f"file size (GB, MB, KB): {get_file_size(parm['fn'])}")
        logging.info(f"d type: {type(d)}")
        logging.info(f"d.keys: {d.keys()}")
        logging.info(f"d['value'].shape: {d['values'].shape}")
        
        ds['ids'].append(id_)
        ds['genes'].append(d['genes'])
        ds['values'].append(d['values'])
    except Exception as e:
        logging.error(f"Error! {e}")
        continue

logging.info(f"-"*50)
memory_info = get_memory_usage()
logging.info(f"... merging start")
logging.info(f"Mem Total: {memory_info['total']} GB")
logging.info(f"Mem Available: {memory_info['available']} GB")
logging.info(f"Mem Used: {memory_info['used']} GB")
logging.info(f"Mem Free: {memory_info['free']} GB")
logging.info(f"Mem Percent: {memory_info['percent']} %")
merged_tokenized_data = {
    "genes": torch.cat(ds['genes'], dim=0),
    "values": torch.cat(ds['values'], dim=0)
}
logging.info(f"-"*50)
logging.info(f"... merging end")
logging.info(f"Mem Total: {memory_info['total']} GB")
logging.info(f"Mem Available: {memory_info['available']} GB")
logging.info(f"Mem Used: {memory_info['used']} GB")
logging.info(f"Mem Free: {memory_info['free']} GB")
logging.info(f"Mem Percent: {memory_info['percent']} %")

num_ds = len(ds['values'])
logging.info(f"="*50)
logging.info(f"merged_tokenized_data type: {type(merged_tokenized_data)}")
torch.save(merged_tokenized_data, f"{save_dir}/cre329_tokenized_merged_numds{num_ds}.pt")
logging.info(f"Finished.")
