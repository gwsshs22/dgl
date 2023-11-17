# Running on node3

MY_RANK=3

for NUM_PARTS in "3" "2"
do

for GRAPH_NAME in "fb10b"
do

python -m omega.tools.partition_facebook_graph \
  --dataset $GRAPH_NAME \
  --input_dir /mydata/gwkim/fb_raw_data/$GRAPH_NAME \
  --num_parts $NUM_PARTS \
  --output $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size 1024 \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-1024

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size 1024 \
  --sampled \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-1024-sampled


python -m omega.tools.partition_facebook_graph \
  --dataset $GRAPH_NAME \
  --input_dir /mydata/gwkim/fb_raw_data/$GRAPH_NAME \
  --num_parts $NUM_PARTS \
  --include_out_edges \
  --output $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS-outedges

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS-outedges/$GRAPH_NAME.json \
  --batch_size 1024 \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-1024

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS-outedges/$GRAPH_NAME.json \
  --batch_size 1024 \
  --sampled \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-1024-sampled

python $DGL_HOME/exp_scripts/distribute_data.py \
  --dataset $GRAPH_NAME \
  --num_parts $NUM_PARTS \
  --my_rank $MY_RANK \
  --machine_ips "node0,node1,node2,node3"

rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-1024/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-1024/
rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-1024-sampled/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-bs-1024-sampled/
rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-1024/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-1024/
rsync -avP --mkpath $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-1024-sampled/ gwkim@node0:$DGL_DATA_HOME/omega_traces/$GRAPH_NAME-random-$NUM_PARTS-outedges-bs-1024-sampled/

python -m omega.tools.compute_degrees \
  --dataset $GRAPH_NAME \
  --data_dir $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS

python -m omega.tools.compute_degrees \
  --dataset $GRAPH_NAME \
  --data_dir $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS-outedges

python $DGL_HOME/exp_scripts/distribute_degrees.py \
  --dataset $GRAPH_NAME \
  --num_parts $NUM_PARTS \
  --my_rank $MY_RANK \
  --machine_ips "node0,node1,node2,node3"

done
done
