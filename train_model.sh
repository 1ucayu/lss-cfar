#!/bin/bash

# Training script for the LSSL model

# Set default values for hyperparameters (you can change these if needed)
DATASET_PATH="/data/lucayu/lss-cfar/dataset/lucacx_corridor_2024-08-27"
NUM_LAYERS=4
HIDDEN_DIM=256 #142
ORDER=256
DT_MIN=1e-3
DT_MAX=8e-5
CHANNELS=1
DROPOUT=0.1
LEARNING_RATE=1e-2
BATCH_SIZE=4
NUM_WORKERS=4
TOTAL_STEPS=10000
WEIGHT_DECAY=1e-1
OPTIMIZER="AdamW"
STEP_SIZE=300
GAMMA=0.5
SAVE_DIR="./checkpoints"
VISUALIZATION_STRIDE=100
GPUS="0"
LOG_DIR="./logs"
LOSS_TYPE="bce"

# Run the training script
python train_model.py \
  --dataset_path $DATASET_PATH \
  --num_layers $NUM_LAYERS \
  --hidden_dim $HIDDEN_DIM \
  --order $ORDER \
  --dt_min $DT_MIN \
  --dt_max $DT_MAX \
  --channels $CHANNELS \
  --dropout $DROPOUT \
  --learning_rate $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --total_steps $TOTAL_STEPS \
  --weight_decay $WEIGHT_DECAY \
  --optimizer $OPTIMIZER \
  --step_size $STEP_SIZE \
  --gamma $GAMMA \
  --save_dir $SAVE_DIR \
  --visualization_stride $VISUALIZATION_STRIDE \
  --gpus $GPUS \
  --log_dir $LOG_DIR \
  --loss_type $LOSS_TYPE