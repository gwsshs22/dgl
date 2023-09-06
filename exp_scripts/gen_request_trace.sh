GRAPH_NAME=$1
NUM_PARTS=$2
BATCH_SIZE=$3

python $DGL_HOME/omega/gen_request_trace.py \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size $BATCH_SIZE \
  --num_traces 100 \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE

python $DGL_HOME/omega/gen_request_trace.py \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size $BATCH_SIZE \
  --num_traces 100 \
  --sampled \
  --fanout 25 \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE-sampled
