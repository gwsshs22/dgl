RESULT_ROOT_DIR=/home/gwkim/omega_trained_models/1018_papers_trans_ns_12_12_12

LOCAL_RANK=3

for GRAPH_NAME in "ogbn-papers100M"
do

for LR in "0.001" "0.01" "0.0001"
do

HIDDEN_DIMS=512
DROPOUT=0.1

echo $GRAPH_NAME $HIDDEN_DIMS $DROPOUT $LR

GNN=gcn
python -m omega.training.train_v2 --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/nl_2_lr_${LR}_do_${DROPOUT}_norm_both --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 10000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --sampling_method ns \
  --num_layers 2 \
  --fanouts 0,0 \
  --gcn_norm both \
  --dropout $DROPOUT \
  --local_rank $LOCAL_RANK

GNN=sage
python -m omega.training.train_v2 --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/nl_3_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 10000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --sampling_method ns \
  --fanouts 12,12,12 \
  --gat_heads 8,8,8 \
  --num_layers 3 \
  --dropout $DROPOUT \
  --local_rank $LOCAL_RANK

python -m omega.training.train_v2 --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/nl_2_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 10000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --sampling_method ns \
  --fanouts 12,12 \
  --gat_heads 8,8 \
  --num_layers 2 \
  --dropout $DROPOUT \
  --local_rank $LOCAL_RANK

GNN=gat
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python -m omega.training.train_v2 --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/nl_3_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 10000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --sampling_method ns \
  --fanouts 12,12,12 \
  --gat_heads 8,8,8 \
  --num_layers 3 \
  --dropout $DROPOUT \
  --local_rank $LOCAL_RANK

PYTORCH_NO_CUDA_MEMORY_CACHING=1 python -m omega.training.train_v2 --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/nl_2_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 10000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --sampling_method ns \
  --fanouts 12,12 \
  --gat_heads 8,8 \
  --num_layers 2 \
  --dropout $DROPOUT \
  --local_rank $LOCAL_RANK

done
done
