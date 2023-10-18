RESULT_ROOT_DIR=/home/gwkim/omega_trained_models/1017_yelp_gcn2

LOCAL_RANK=3

for GRAPH_NAME in "yelp"
do

for LR in "0.01" "0.001" "0.0001"
do

HIDDEN_DIMS=512
DROPOUT=0.1

echo $GRAPH_NAME $HIDDEN_DIMS $DROPOUT $LR

GNN=gcn2
for NUM_LAYERS in "8" "6" "4" "2"
do
python -m omega.training.train_v2 --result_dir $RESULT_ROOT_DIR/$GRAPH_NAME/${GNN}/nl_${NUM_LAYERS}_lr_${LR}_do_${DROPOUT}_norm_both --lr $LR \
  --graph_name $GRAPH_NAME --num_epochs 10000 \
  --saint_data_root $DGL_RAW_DATA_HOME/saint_datasets \
  --ogbn_data_root $DGL_RAW_DATA_HOME/ogbn_datasets \
  --gnn $GNN \
  --num_hiddens $HIDDEN_DIMS \
  --sampling_method full \
  --num_layers $NUM_LAYERS \
  --gcn2_alpha 0.5 \
  --gcn_norm both \
  --dropout $DROPOUT \
  --local_rank $LOCAL_RANK
done
done
done
