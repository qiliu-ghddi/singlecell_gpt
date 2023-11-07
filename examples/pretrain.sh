#!/bin/bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"

NPROC=2

QUERY_NAME="demo_preprocessed_id8"
WORK_DIR="/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt"
DATASET=${WORK_DIR}"/data/preprocessed/all_counts"
JOB_NAME="cellxgene_census_${QUERY_NAME}"
SAVE_DIR=${WORK_DIR}"/save/${QUERY_NAME}"
VOCAB_PATH=${WORK_DIR}"/data/default_census_vocab.json"

LOG_INTERVAL=2000
VALID_SIZE_OR_RATIO=0.2
MAX_LENGTH=1200
per_proc_batch_size=8
LAYERS=12
MODEL_SCALE=8
NHEADS=8
EPOCHS=16
LR=0.0001

python pretrain.py \
    --data-source $DATASET \
    --save-dir ${SAVE_DIR}/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
    --vocab-path ${VOCAB_PATH} \
    --valid-size-or-ratio $VALID_SIZE_OR_RATIO \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $(($per_proc_batch_size * 2)) \
    --nlayers $LAYERS \
    --nheads $NHEADS \
    --embsize $((MODEL_SCALE * 64)) \
    --d-hid $((MODEL_SCALE * 64)) \
    --grad-accu-steps 1 \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup-ratio-or-step 10000 \
    --log-interval $LOG_INTERVAL \
    --save-interval $(($LOG_INTERVAL * 3)) \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 |
    awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'

