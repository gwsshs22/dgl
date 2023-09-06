#!/bin/bash

GRAPH_NAME=$1
NUM_PARTS=$2

python $DGL_HOME/omega/partition_graph.py \
  --dataset $GRAPH_NAME \
  --num_parts $NUM_PARTS \
  --part_method random \
  --output $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS
