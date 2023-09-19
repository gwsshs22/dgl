RESULT_ROOT_DIR=/home/gwkim/omega_trained_models/0918

for GRAPH_NAME in "ogbn-products" "reddit"
do

for GNN in "gcn" "sage" "gat"
do

for LR in "0.01" "0.001" "0.0001"
do

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/0_0_lr_${LR} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens 128 \
  --gat_heads 8,8 \
  --num_layers 2 \
  --local_rank 1

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/10_25_${LR} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens 128 \
  --gat_heads 8,8 \
  --num_layers 2 \
  --fanouts 10,25 \
  --local_rank 1

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/0_0_0_lr_${LR} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens 128 \
  --gat_heads 8,8,8 \
  --num_layers 3 \
  --local_rank 1

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/5_10_15_${LR} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens 128 \
  --gat_heads 8,8,8 \
  --num_layers 3 \
  --fanouts 5,10,15 \
  --local_rank 1

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/0_0_0_0_lr_${LR} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens 128 \
  --gat_heads 8,8,8,8 \
  --num_layers 4 \
  --local_rank 1

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/5_10_15_20_${LR} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens 128 \
  --gat_heads 8,8,8,8 \
  --num_layers 4 \
  --fanouts 5,10,15,20 \
  --local_rank 1

done
done
done