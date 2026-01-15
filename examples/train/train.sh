#!/bin/bash
set +x

# export HF_ENDPOINT="https://hf-mirror.com"

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_rlvr_socioseg_pipeline.py --config_path $CONFIG_PATH  --config_name rlvr_megatron