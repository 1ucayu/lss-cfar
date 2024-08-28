#!/bin/bash

# Training script for the LSSL model

# Define dataset paths and calibration paths as arrays
DATASET_PATHS=(
    "/data/lucayu/lss-cfar/dataset/cx_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/dataset/cx_env_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/dataset/luca_env_hw_101_2024-08-23"
    "/data/lucayu/lss-cfar/dataset/luca_hw_101_2024-08-23"
    "/data/lucayu/lss-cfar/dataset/lucacx_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/dataset/lucacx_env_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/dataset/wayne_env_office_2024-08-27"
    "/data/lucayu/lss-cfar/dataset/wayne_office_2024-08-27"
)

CALIBRATION_PATHS=(
    "/data/lucayu/lss-cfar/raw_dataset/cx_env_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/raw_dataset/cx_env_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/raw_dataset/luca_env_hw_101_2024-08-23"
    "/data/lucayu/lss-cfar/raw_dataset/luca_env_hw_101_2024-08-23"
    "/data/lucayu/lss-cfar/raw_dataset/lucacx_env_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/raw_dataset/lucacx_env_corridor_2024-08-27"
    "/data/lucayu/lss-cfar/raw_dataset/wayne_env_office_2024-08-27"
    "/data/lucayu/lss-cfar/raw_dataset/wayne_env_office_2024-08-27"
)

# Run the training script and pass arrays as arguments
python train_model.py \
  --dataset_paths "${DATASET_PATHS[@]}" \
  --calibration_paths "${CALIBRATION_PATHS[@]}" \
  --num_layers 4 \
  --hidden_dim 256 \
  --order 256 \
  --dt_min 1e-3 \
  --dt_max 8e-5 \
  --channels 1 \
  --dropout 0.0 \
  --learning_rate 1e-2 \
  --batch_size 4 \
  --num_workers 4 \
  --total_steps 10000 \
  --weight_decay 1e-1 \
  --optimizer AdamW \
  --step_size 300 \
  --gamma 0.5 \
  --save_dir "./checkpoints" \
  --visualization_stride 100 \
  --gpus 0 \
  --log_dir "./logs" \
  --loss_type "l1"
