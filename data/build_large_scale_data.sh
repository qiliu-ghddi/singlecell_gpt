#!/bin/bash

# conda activate /home/qiliu02/miniconda3/envs/flash-attn

WORK_DIR="/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/release"
INPUT_DIR=${WORK_DIR}"/data/raw"
OUTPUT_DIR=${WORK_DIR}"/data/preprocessed"
VOCAB_FILE=${WORK_DIR}"/data/default_census_vocab.json"

include_ids="8"

python build_large_scale_data.py --input-dir ${INPUT_DIR} --output-dir ${OUTPUT_DIR} --vocab-file ${VOCAB_FILE} --include-ids ${include_ids}


