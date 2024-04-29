RESULT_ROOT_DIR=/nfs/gwkim/omega_nsdi25/models/240428_sage_lstm

LOCAL_RANK=3

for GRAPH_NAME in "yelp"
do

for LR in "0.01" "0.001" "0.0001"
do

HIDDEN_DIMS=512
DROPOUT=0.1
NUM_EPOCHS=30

echo $GRAPH_NAME $HIDDEN_DIMS $DROPOUT $LR

GNN=sage
python -m omega.training.train_v2 --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/nl_3_lr_${LR}_do_${DROPOUT} --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs $NUM_EPOCHS --patience $NUM_EPOCHS \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --sampling_method ns \
  --fanouts 5,10,15 \
  --gat_heads 8,8,8 \
  --num_layers 3 \
  --dropout $DROPOUT \
  --sage_aggr lstm \
  --local_rank $LOCAL_RANK

done
done
