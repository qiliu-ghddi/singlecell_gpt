#!/bin/bash

WORK_DIR="/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/release"
QUERY_NAME="cellgene20230606_min_counts60_ds_id8"
DATA_SOURCE=${WORK_DIR}"/data/preprocessed/all_counts"
VOCAB_PATH=${WORK_DIR}"/data/default_census_vocab.json"
INPUT_STYLE="binned"
INPUT_EMB_STYLE="continuous"
SAVE_DIR=${WORK_DIR}"/save/${QUERY_NAME}"

N_BINS=51

python process_allcounts.py --data-source ${DATA_SOURCE} \
    --input-style ${INPUT_STYLE} \
    --input-emb-style ${INPUT_EMB_STYLE} \
    --save-dir ${SAVE_DIR} \
    --vocab-path ${VOCAB_PATH} \
    --n-bins ${N_BINS}
