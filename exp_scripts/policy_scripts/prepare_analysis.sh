#!/bin/bash

RAW_DATA_ROOT=/home/gwkim/omega_raw_datasets
ANALYSIS_ROOT=/home/gwkim/omega_analysis

for GRAPH_NAME in ogbn-papers100M amazon ogbn-products reddit yelp
do
  echo $GRAPH_NAME
  python -m omega.tools.partition_graph \
    --dataset $GRAPH_NAME \
    --num_parts 1 \
    --part_method random \
    --ogbn_data_root $RAW_DATA_ROOT/ogbn_datasets \
    --saint_data_root $RAW_DATA_ROOT/saint_datasets \
    --num_parts 1 \
    --infer_prob 0.25 \
    --rel_to_tests \
    --output $ANALYSIS_ROOT/datasets/$GRAPH_NAME

  python -m omega.tools.gen_request_trace \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --batch_size 1024 \
    --output $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024

  python -m omega.tools.gen_request_trace \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --batch_size 1024 \
    --sampled \
    --fanout 25 \
    --output $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled
done
