# Running on node0

NUM_PARTS=4
for GRAPH_NAME in "reddit" "yelp" "amazon" "ogbn-products" "ogbn-papers100M"
do

python -m omega.tools.compute_degrees \
  --dataset $GRAPH_NAME \
  --data_dir $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS

python -m omega.tools.compute_degrees \
  --dataset $GRAPH_NAME \
  --data_dir $DGL_DATA_HOME/omega_datasets-$NUM_PARTS/$GRAPH_NAME-random-$NUM_PARTS-outedges

python $DGL_HOME/exp_scripts/distribute_degrees.py \
  --dataset $GRAPH_NAME \
  --num_parts 4 \
  --my_rank 0 \
  --machine_ips "node0,node1,node2,node3"

done
