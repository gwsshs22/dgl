#!/bin/bash

EXP_ROOT_DIR=$1
EXTRA_ENV_NAME=$2

python $DGL_HOME/exp_scripts/run_overall_latency.py --exp_root_dir $EXP_ROOT_DIR --extra_env_names "$EXTRA_ENV_NAME"

python $DGL_HOME/exp_scripts/run_layers.py --exp_root_dir $EXP_ROOT_DIR --extra_env_names "$EXTRA_ENV_NAME"

python $DGL_HOME/exp_scripts/run_batch_sizes.py --exp_root_dir $EXP_ROOT_DIR --extra_env_names "$EXTRA_ENV_NAME"

python $DGL_HOME/exp_scripts/run_features.py --exp_root_dir $EXP_ROOT_DIR --extra_env_names "$EXTRA_ENV_NAME"

python $DGL_HOME/exp_scripts/run_hidden_dims.py --exp_root_dir $EXP_ROOT_DIR --extra_env_names "$EXTRA_ENV_NAME"

python $DGL_HOME/exp_scripts/run_scalability.py --exp_root_dir $EXP_ROOT_DIR --extra_env_names "$EXTRA_ENV_NAME"

