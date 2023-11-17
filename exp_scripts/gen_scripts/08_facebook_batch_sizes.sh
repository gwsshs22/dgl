# Running on node3

MY_RANK=3
NUM_PARTS=4

for GRAPH_NAME in "fb10b" "fb5b"
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

rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE/
rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE-sampled/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-$BATCH_SIZE-sampled/
rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-$BATCH_SIZE/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-$BATCH_SIZE/
rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-$BATCH_SIZE-sampled/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-$BATCH_SIZE-sampled/

done
done