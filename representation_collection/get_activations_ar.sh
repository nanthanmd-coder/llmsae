#!/usr/bin/env bash

seed=${1:-55}
model_path=${2:-"../pretrained_models/llava-v1.5-7b"}
arrow_dir=${3:-"../../CC3M/cc3m-wds"}
arrow_pattern=${4:-"cc3m-wds-train-*.arrow"}
save_path=${5:-"./activations"}

target_layer_name=${6:-"model.layers.30"}
chunk_idx=${7:-0}
num_chunks=${8:-1}

CUDA_VISIBLE_DEVICES=1 python activation_collector_ar.py \
  --model-path "${model_path}" \
  --arrow_dir "${arrow_dir}" \
  --arrow_pattern "${arrow_pattern}" \
  --seed "${seed}" \
  --save_path "${save_path}" \
  --target_layer_name "${target_layer_name}" \
  --chunk-idx "${chunk_idx}" \
  --num-chunks "${num_chunks}"