RESULT_ROOT_DIR=/home/gwkim/omega_trained_models/1004_1
GRAPH_NAME=yelp
DROPOUT=0.1
HIDDEN_DIMS=512

for GNN in "gcn"
do

for LR in "0.01" "0.001" "0.0001"
do

echo $GRAPH_NAME $HIDDEN_DIMS

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/10_25_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --gat_heads 8,8 \
  --fanouts 10,25 \
  --num_layers 2 \
  --dropout $DROPOUT \
  --local_rank 0

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/5_10_15_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --gat_heads 8,8,8 \
  --fanouts 5,10,15 \
  --num_layers 3 \
  --dropout $DROPOUT \
  --local_rank 0

python -m omega.training.train --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/5_10_15_20_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 5000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --gat_heads 8,8,8,8 \
  --fanouts 5,10,15,20 \
  --num_layers 4 \
  --dropout $DROPOUT \
  --local_rank 0

done
done
