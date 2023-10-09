ANALYSIS_ROOT=/home/gwkim/omega_analysis

# 2-Layers
TRAINING_DIR=/home/gwkim/omega_trained_models/0928/yelp/gat/10_25_0.001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_gat_2
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1

TRAINING_DIR=/home/gwkim/omega_trained_models/1004_1/yelp/gcn/10_25_lr_0.001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_gcn_2
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1

TRAINING_DIR=/home/gwkim/omega_trained_models/0928/yelp/sage/10_25_0.001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_sage_2
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1


# 3-Layers
TRAINING_DIR=/home/gwkim/omega_trained_models/0928/yelp/gat/5_10_15_0.001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_gat_3
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1

TRAINING_DIR=/home/gwkim/omega_trained_models/1004_1/yelp/gcn/5_10_15_lr_0.0001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_gcn_3
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1

TRAINING_DIR=/home/gwkim/omega_trained_models/0928/yelp/sage/5_10_15_0.001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_sage_3
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1


# 4-Layers
TRAINING_DIR=/home/gwkim/omega_trained_models/0928/yelp/gat/5_10_15_20_0.001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_gat_4
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1

TRAINING_DIR=/home/gwkim/omega_trained_models/1004_1/yelp/gcn/5_10_15_20_lr_0.0001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_gcn_4
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1

TRAINING_DIR=/home/gwkim/omega_trained_models/0928/yelp/sage/5_10_15_20_0.0001_do_0.1
GRAPH_NAME=yelp
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/yelp_sage_4
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank 1
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank 1

