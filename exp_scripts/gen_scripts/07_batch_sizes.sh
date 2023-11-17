# Running on node0

MY_RANK=0
NUM_PARTS=4

for GRAPH_NAME in "ogbn-products" "yelp" "amazon" "reddit" "ogbn-papers100M"
do
for BATCH_SIZE in "128" "256" "512" "2048"
do

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size $BATCH_SIZE \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size $BATCH_SIZE \
  --sampled \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE-sampled

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS-outedges/$GRAPH_NAME.json \
  --batch_size $BATCH_SIZE \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-$BATCH_SIZE

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS-outedges/$GRAPH_NAME.json \
  --batch_size $BATCH_SIZE \
  --sampled \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-$BATCH_SIZE-sampled

done
done