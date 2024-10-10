#!/bin/sh
nvcc --version
python -c "import torch; print(torch.version.cuda)"

NPROC=2
# DATASET="path_to/datasets/3faad104-2ab8-4434-816d-474d8d2641db.scb"
# JOB_NAME="cellxgene_3faad1"
# LOG_INTERVAL=1000
# VALID_SIZE_OR_RATIO=0.1

QUERY_NAME="cellgene20230606_ds1_min_counts60"
WORK_DIR="/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/"
DATASET="/home/qiliu02/GHDDI/DS-group/ghddixcre_singlecell_gpt/data/${QUERY_NAME}/all_counts"
JOB_NAME="cellxgene_census_${QUERY_NAME}"

# QUERY_NAME="pan-cancer"
# DATASET="/scratch/ssd004/datasets/cellxgene/scb_strict/${QUERY_NAME}/all_counts"
# JOB_NAME="cellxgene_census_${QUERY_NAME}"


LOG_INTERVAL=2000
VALID_SIZE_OR_RATIO=0.3
# VALID_SIZE_OR_RATIO=0.03
MAX_LENGTH=1200
per_proc_batch_size=32
LAYERS=12
MODEL_SCALE=8
NHEADS=8
EPOCHS=6
LR=0.0001

SAVE_DIR=${WORK_DIR}"/save/${QUERY_NAME}"
VOCAB_PATH=${WORK_DIR}"/data/default_census_vocab.json"

python -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    pretrain.py \
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


# SAVE_DIR="/scratch/ssd004/datasets/cellxgene/save"
# VOCAB_PATH="path_to/tokenizer/default_census_vocab.json"

# python_ -m torch.distributed.launch \
#     --nproc_per_node=$NPROC \
#     pretrain.py \
#     --data-source $DATASET \
#     --save-dir ./save/$JOB_NAME-$(date +%b%d-%H-%M-%Y) \
#     --vocab-path ${VOCAB_PATH} \
#     --valid-size-or-ratio $VALID_SIZE_OR_RATIO \
#     --max-seq-len $MAX_LENGTH \
#     --batch-size $per_proc_batch_size \
#     --eval-batch-size $(($per_proc_batch_size * 2)) \
#     --nlayers $LAYERS \
#     --nheads 8 \
#     --embsize $((MODEL_SCALE * 64)) \
#     --d-hid $((MODEL_SCALE * 64)) \
#     --grad-accu-steps 1 \
#     --epochs 6 \
#     --lr 0.0001 \
#     --warmup-ratio-or-step 10000 \
#     --log-interval $LOG_INTERVAL \
#     --save-interval $(($LOG_INTERVAL * 3)) \
#     --trunc-by-sample \
#     --no-cls \
#     --no-cce \
#     --fp16 |
#     awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
