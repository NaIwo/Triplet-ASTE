DATA_PATH="/home/iwo/Pulpit/Studia/Triplet/ASTE/dataset/data/ASTE_data_v2"
RESULTS_SAVE_PATH="/home/iwo/Pulpit/Studia/Triplet/ASTE/experiments/experiment_results/"
MODEL_CHECKPOINT_PATH="../models/aste_model"

DATASET_NAME="14lap"
for id in 0 1 2 3 4 5
do
  $(which python) model_trainer.py \
        --data_path DATA_PATH \
        --dataset_name $VALID_PATH \
        --result_path RESULTS_SAVE_PATH \
        --model_checkpoint_path $MODEL_CHECKPOINT_PATH \
        --id $id
done


DATASET_NAME="14res"
for id in 0 1 2 3 4 5
do
  $(which python) model_trainer.py \
        --data_path DATA_PATH \
        --dataset_name $VALID_PATH \
        --result_path RESULTS_SAVE_PATH \
        --model_checkpoint_path $MODEL_CHECKPOINT_PATH \
        --id $id
done


DATASET_NAME="15res"
for id in 0 1 2 3 4 5
do
  $(which python) model_trainer.py \
        --data_path DATA_PATH \
        --dataset_name $VALID_PATH \
        --result_path RESULTS_SAVE_PATH \
        --model_checkpoint_path $MODEL_CHECKPOINT_PATH \
        --id $id
done

DATASET_NAME="16res"
for id in 0 1 2 3 4 5
do
  $(which python) model_trainer.py \
        --data_path DATA_PATH \
        --dataset_name $VALID_PATH \
        --result_path RESULTS_SAVE_PATH \
        --model_checkpoint_path $MODEL_CHECKPOINT_PATH \
        --id $id
done
