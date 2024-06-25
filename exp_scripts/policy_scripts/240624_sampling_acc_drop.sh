ANALYSIS_ROOT=/nfs/gwkim/omega_nsdi25
OUTPUT_ROOT_DIR=/nfs/gwkim/omega_nsdi25/policy_results/20240624_sampled

# AMAZON
GRAPH_NAME=amazon
TRAINING_DIR=/nfs/gwkim/omega_nsdi25/models/240418/amazon/gcn/nl_2_lr_0.001_do_0.1_norm_both
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gcn_2
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/dgl_datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1


GRAPH_NAME=amazon
TRAINING_DIR=/nfs/gwkim/omega_nsdi25/models/240418/amazon/gat/nl_3_lr_0.001_do_0.1
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gat_3
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/dgl_datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1

# YELP
GRAPH_NAME=yelp
TRAINING_DIR=/nfs/gwkim/omega_nsdi25/models/240418/yelp/gcn/nl_2_lr_0.01_do_0.1_norm_both
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gcn_2
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/dgl_datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1

GRAPH_NAME=yelp
TRAINING_DIR=/nfs/gwkim/omega_nsdi25/models/240418/yelp/gat/nl_3_lr_0.001_do_0.1
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gat_3
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/dgl_datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1
