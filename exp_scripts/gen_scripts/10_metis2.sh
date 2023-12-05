# Running on node0
MY_RANK=0
NUM_PARTS=4
GRAPH_NAME=amazon
PART_METHOD=metis

python -m omega.tools.partition_graph \
  --dataset $GRAPH_NAME \
  --num_parts $NUM_PARTS \
  --part_method $PART_METHOD \
  --infer_prob 0.25 \
  --rel_to_tests \
  --ogbn_data_root $DGL_DATA_HOME/omega_raw_datasets/ogbn_datasets \
  --saint_data_root $DGL_DATA_HOME/omega_raw_datasets/saint_datasets \
  --output $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size 1024 \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-bs-1024

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS/$GRAPH_NAME.json \
  --batch_size 1024 \
  --sampled \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-bs-1024-sampled

python -m omega.tools.partition_graph \
  --dataset $GRAPH_NAME \
  --num_parts $NUM_PARTS \
  --part_method $PART_METHOD \
  --infer_prob 0.25 \
  --rel_to_tests \
  --ogbn_data_root $DGL_DATA_HOME/omega_raw_datasets/ogbn_datasets \
  --saint_data_root $DGL_DATA_HOME/omega_raw_datasets/saint_datasets \
  --include_out_edges \
  --output $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-outedges

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-outedges/$GRAPH_NAME.json \
  --batch_size 1024 \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-outedges-bs-1024

python -m omega.tools.gen_request_trace \
  --graph_name $GRAPH_NAME \
  --part_config $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-outedges/$GRAPH_NAME.json \
  --batch_size 1024 \
  --sampled \
  --output $DGL_DATA_HOME/omega_traces/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-outedges-bs-1024-sampled

python -m omega.tools.compute_degrees \
  --dataset $GRAPH_NAME \
  --data_dir $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS

python -m omega.tools.compute_degrees \
  --dataset $GRAPH_NAME \
  --data_dir $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-$PART_METHOD-$NUM_PARTS-outedges

python $DGL_HOME/exp_scripts/distribute_data.py \
  --dataset $GRAPH_NAME \
  --num_parts 4 \
  --graph_partitioning metis \
  --my_rank 0 \
  --machine_ips "node0,node1,node2,node3"

python $DGL_HOME/exp_scripts/distribute_degrees.py \
  --dataset $GRAPH_NAME \
  --num_parts 4 \
  --graph_partitioning metis \
  --my_rank 0 \
  --machine_ips "node0,node1,node2,node3"