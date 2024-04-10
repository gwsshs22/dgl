ANALYSIS_ROOT=/home/gwkim/omega_osdi/omega_analysis
OUTPUT_ROOT_DIR=/home/gwkim/omega_osdi/nsdi25/approx_errors/0410_yelp

GRAPH_NAME=yelp

TRAINING_DIR=/home/gwkim/omega_osdi/omega_trained_models/1017/yelp/gat/nl_3_lr_0.001_do_0.1
OUTPUT_DIR=$OUTPUT_ROOT_DIR/${GRAPH_NAME}_gat_3
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.approx_error_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 0
