DATA_PATH="/home/iwo/Pulpit/Studia/Triplet/ASTE/dataset/data/ASTE_data_v2"
DATASET_NAME="14lap"
NEPTUNE_API_KEY="ANONYMOUS"
MODEL_CHECKPOINT_PATH="../models/aste_model"

$(which python) model_trainer.py \
      --data_path DATA_PATH \
      --dataset_name $VALID_PATH \
      --api_key $NEPTUNE_API_KEY \
      --model_checkpoint_path $MODEL_CHECKPOINT_PATH
