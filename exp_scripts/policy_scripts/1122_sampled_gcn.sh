ANALYSIS_ROOT=/home/gwkim/omega_analysis
OUTPUT_ROOT_DIR=/home/gwkim/omega_policy_analysis

# AMAZON
GRAPH_NAME=amazon
TRAINING_DIR=/home/gwkim/omega_trained_models/1017/amazon/gcn/nl_2_lr_0.01_do_0.5_norm_both
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gcn_2
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1

# OGBN_PRODUCTS
GRAPH_NAME=ogbn-products
TRAINING_DIR=/home/gwkim/omega_trained_models/1017/ogbn-products/gcn/nl_2_lr_0.0001_do_0.5_norm_both
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gcn_2
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1

# OGBN_PAPERS100M
GRAPH_NAME=ogbn-papers100M
TRAINING_DIR=/home/gwkim/omega_trained_models/1017_papers_trans/ogbn-papers100M/gcn/nl_2_lr_0.001_do_0.5_norm_both
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gcn_2
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1

# REDDIT
GRAPH_NAME=reddit
TRAINING_DIR=/home/gwkim/omega_trained_models/1017/reddit/gcn/nl_2_lr_0.01_do_0.5_norm_both
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gcn_2
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1

# YELP
GRAPH_NAME=yelp
TRAINING_DIR=/home/gwkim/omega_trained_models/1017/yelp/gcn/nl_2_lr_0.01_do_0.1_norm_both
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gcn_2
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024-sampled --sampled \
    --output_dir ${OUTPUT_DIR}_sampled --local_rank 1
