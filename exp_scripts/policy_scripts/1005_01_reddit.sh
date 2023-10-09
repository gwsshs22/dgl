ANALYSIS_ROOT=/home/gwkim/omega_analysis
LOCAL_RANK=1
# 2-Layers
TRAINING_DIR=/home/gwkim/omega_trained_models/0918/reddit/gat/10_25_0.001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_gat_2
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK

TRAINING_DIR=/home/gwkim/omega_trained_models/1004_2/reddit/gcn/10_25_0.001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_gcn_2
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK

TRAINING_DIR=/home/gwkim/omega_trained_models/0918/reddit/sage/10_25_0.0001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_sage_2
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK


# 3-Layers
TRAINING_DIR=/home/gwkim/omega_trained_models/0918/reddit/gat/5_10_15_0.001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_gat_3
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK

TRAINING_DIR=/home/gwkim/omega_trained_models/1004_2/reddit/gcn/5_10_15_0.001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_gcn_3
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK

TRAINING_DIR=/home/gwkim/omega_trained_models/0918/reddit/sage/5_10_15_0.001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_sage_3
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK


# 4-Layers
TRAINING_DIR=/home/gwkim/omega_trained_models/0918/reddit/gat/5_10_15_20_0.001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_gat_4
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK

TRAINING_DIR=/home/gwkim/omega_trained_models/1004_2/reddit/gcn/5_10_15_20_0.001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_gcn_4
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK

TRAINING_DIR=/home/gwkim/omega_trained_models/0918/reddit/sage/5_10_15_20_0.0001
GRAPH_NAME=reddit
OUTPUT_DIR=/home/gwkim/omega_policy_analysis/reddit_sage_4
python -m omega.tools.gen_precoms \
  --graph_name $GRAPH_NAME \
  --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
  --training_dir $TRAINING_DIR \
  --local_rank $LOCAL_RANK
python -m omega.tools.recompute_policy_analysis \
    --graph_name $GRAPH_NAME \
    --part_config $ANALYSIS_ROOT/datasets/$GRAPH_NAME/$GRAPH_NAME.json \
    --training_dir $TRAINING_DIR \
    --trace_dir $ANALYSIS_ROOT/traces/$GRAPH_NAME-random-1024 \
    --output_dir $OUTPUT_DIR --local_rank $LOCAL_RANK
