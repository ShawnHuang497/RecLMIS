#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
gpu=$1
config_path=$2

export PYTHONPATH=$ROOT:$PYTHONPATH


echo ${gpu}

CUDA_VISIBLE_DEVICES="${gpu}" python ./train_model.py -c ${config_path}